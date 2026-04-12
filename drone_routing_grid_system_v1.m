%% DECENTRALIZED DRONE DELIVERY SIMULATION (SF GRID VERSION)
% Adapted to match the style of your baseline grid simulation while adding:
%   1) ft-based geometry
%   2) 400 ft max altitude constraint
%   3) building-height exclusion with 5 ft clearance buffer
%   4) time-series demand arrivals
%   5) conflict-aware route reservation in space-time
%   6) direction-based altitude lanes (N/S/E/W)
%   7) intersection capacity checks and launch delay if needed
%
% IMPORTANT
% ---------
% This version is ready to run as-is with a synthetic building map.
% To use real SF data later, replace the synthetic 'buildingHeightFt' matrix
% with a raster/grid derived from a GIS layer or building-height dataset.
%
% The script assumes drones move node-to-node on a rectangular grid.
% Nodes are 20 ft apart, as requested.

clear; clc; close all;

%% ===================== 1. PARAMETERS ==================================

% ---------------- Grid ----------------
gridRows   = 30;         % number of rows of nodes
gridCols   = 30;         % number of cols of nodes
gridStepFt = 20;         % distance between neighboring nodes in feet
nNodes     = gridRows * gridCols;

% ---------------- Time ----------------
dt = 1;                  % seconds per simulation step
T  = 900;                % total simulation time [s]
timeVec = 0:dt:T;

% ---------------- Drone physics ----------------
cruiseSpeedFt_s = 20;    % horizontal speed [ft/s]
turnPenalty_s   = 2;     % extra delay for 90-deg turn [s]
takeoff_s       = 4;     % launch overhead [s]
landing_s       = 4;     % landing overhead [s]

% ---------------- Safety / altitude ----------------
maxAltitudeFt   = 400;
clearanceFt     = 5;     % must remain at least this far from buildings
collisionDistFt = 10;    % threshold for collision risk

% Assigned cruise altitude by direction
alt.N = 260;
alt.S = 280;
alt.E = 300;
alt.W = 320;
alt.turn = 340;          % temporary altitude when turning at an intersection

assert(all(cell2mat(struct2cell(alt)) <= maxAltitudeFt), 'Altitude assignment exceeds 400 ft.');

% ---------------- Capacity rules ----------------
maxSameDirPerEdge = 1;   % conservative: one drone per directed edge-time slot
maxTurnsPerNode    = 1;  % max number of simultaneous turn maneuvers per node-time slot
maxNodePassages    = 2;  % total simultaneous node occupancy at a node-time slot
maxLaunchDelay_s   = 180;

% ---------------- Demand ----------------
rng(7);
baseDemandLambda = 0.07;      % jobs/s baseline
peakDemandLambda = 0.20;      % jobs/s peak
peakCenter       = 450;       % peak time [s]
peakWidth        = 120;       % width of peak [s]

% ---------------- Depots / service ----------------
% You can change these to any node IDs you want.
depotNodeIDs = [sub2ind([gridRows, gridCols], 3, 3), ...
                sub2ind([gridRows, gridCols], 3, 28), ...
                sub2ind([gridRows, gridCols], 28, 6), ...
                sub2ind([gridRows, gridCols], 26, 26)];

% ---------------- Building map mode ----------------
useSyntheticBuildings = true;
% If you later have a real raster, set useSyntheticBuildings = false and
% load buildingHeightFt here as a gridRows x gridCols matrix of max building
% height near each node, or derive blocked edges/nodes from your GIS source.

%% ===================== 2. BUILD GRID ==================================

[rowIdx, colIdx] = ndgrid(1:gridRows, 1:gridCols);
nodeX = (colIdx(:)-1) * gridStepFt;
nodeY = (gridRows-rowIdx(:)) * gridStepFt;   % top row has largest y visually

nodeID = @(r,c) sub2ind([gridRows, gridCols], r, c);

% 4-neighbor connectivity
edgeList = zeros((gridRows*(gridCols-1) + gridCols*(gridRows-1))*2, 2);
k = 0;
for r = 1:gridRows
    for c = 1:gridCols
        u = nodeID(r,c);
        if c < gridCols
            v = nodeID(r,c+1);
            k = k + 1; edgeList(k,:) = [u v];
            k = k + 1; edgeList(k,:) = [v u];
        end
        if r < gridRows
            v = nodeID(r+1,c);
            k = k + 1; edgeList(k,:) = [u v];
            k = k + 1; edgeList(k,:) = [v u];
        end
    end
end
edgeList = edgeList(1:k,:);

%% ===================== 3. BUILDING HEIGHT MAP ==========================

buildingHeightFt = zeros(gridRows, gridCols);

if useSyntheticBuildings
    % Create a few clusters of tall buildings to mimic downtown-like regions.
    % Each entry is a node-centered "building height" for route exclusion logic.
    buildingHeightFt(10:16, 12:18) = 220;
    buildingHeightFt(11:15, 13:17) = 330;
    buildingHeightFt(7:11, 21:25)  = 180;
    buildingHeightFt(19:25, 8:12)  = 250;
    buildingHeightFt(20:23, 9:11)  = 360;
    buildingHeightFt(4:6, 7:9)     = 140;
    buildingHeightFt(24:27, 22:26) = 190;
end

% Nodes that are too close to tall buildings for the chosen lane structure.
% Since all cruise altitudes are <= 340 ft, anything >= (alt.turn - clearance)
% is effectively impassable in this simplified grid model.
blockedNode = buildingHeightFt >= (alt.turn - clearanceFt);

% Never block depots completely
for i = 1:numel(depotNodeIDs)
    [rr,cc] = ind2sub([gridRows,gridCols], depotNodeIDs(i));
    blockedNode(rr,cc) = false;
end

% Mark blocked directed edges if either endpoint is blocked.
blockedEdge = false(size(edgeList,1),1);
for e = 1:size(edgeList,1)
    u = edgeList(e,1); v = edgeList(e,2);
    [ru,cu] = ind2sub([gridRows,gridCols],u);
    [rv,cv] = ind2sub([gridRows,gridCols],v);
    if blockedNode(ru,cu) || blockedNode(rv,cv)
        blockedEdge(e) = true;
    end
end

%% ===================== 4. DEMAND TIME SERIES ===========================

lambda_t = baseDemandLambda + ...
    (peakDemandLambda - baseDemandLambda) * exp(-0.5*((timeVec - peakCenter)/peakWidth).^2);

% Poisson arrivals each second
arrivalsPerStep = zeros(size(lambda_t));
for t = 1:length(lambda_t)
    arrivalsPerStep(t) = my_poissrnd(lambda_t(t) * dt);
end

jobs = struct('requestTime', {}, 'origin', {}, 'dest', {}, 'assignedDepot', {});
jobCounter = 0;

feasibleCustomerNodes = find(~blockedNode(:));
feasibleCustomerNodes = setdiff(feasibleCustomerNodes, depotNodeIDs);

for t = 1:numel(timeVec)
    nArr = arrivalsPerStep(t);
    for j = 1:nArr
        jobCounter = jobCounter + 1;
        dest = feasibleCustomerNodes(randi(numel(feasibleCustomerNodes)));
        % decentralized flavor: nearest depot handles request
        dists = manhattanNodeDistance(depotNodeIDs, dest, gridRows, gridCols) * gridStepFt;
        [~,idxMin] = min(dists);
        jobs(jobCounter).requestTime   = timeVec(t);
        jobs(jobCounter).origin        = depotNodeIDs(idxMin);
        jobs(jobCounter).dest          = dest;
        jobs(jobCounter).assignedDepot = depotNodeIDs(idxMin);
    end
end
nJobs = numel(jobs);

fprintf('Generated %d delivery requests over %d seconds.\n', nJobs, T);

%% ===================== 5. RESERVATION TABLES ===========================

% Each reservation table is indexed by time step.
nodeOcc  = cell(T+1,1);      % nodeOcc{t} = node IDs occupied at integer time t-1
turnOcc  = cell(T+1,1);      % turnOcc{t} = node IDs with active turn maneuver
edgeOcc  = cell(T+1,1);      % edgeOcc{t} = directed edge row indices occupied

% Drone/job results
jobStatus        = strings(nJobs,1);
launchDelay_s    = nan(nJobs,1);
flightTime_s     = nan(nJobs,1);
routeNodeLists   = cell(nJobs,1);
routeTimes       = cell(nJobs,1);
routeDirs        = cell(nJobs,1);
routeAlts        = cell(nJobs,1);
routeEdgeRows    = cell(nJobs,1);
collisionRisks   = zeros(nJobs,1);
turnCounts       = zeros(nJobs,1);

%% ===================== 6. PLAN EACH JOB ================================

for j = 1:nJobs
    tReq = jobs(j).requestTime;
    origin = jobs(j).origin;
    dest   = jobs(j).dest;

    foundPlan = false;

    for launchTime = tReq:min(T-1, tReq + maxLaunchDelay_s)
        plan = planRouteTimeAware(origin, dest, launchTime, edgeList, blockedEdge, ...
            blockedNode, gridRows, gridCols, cruiseSpeedFt_s, gridStepFt, ...
            turnPenalty_s, takeoff_s, landing_s, edgeOcc, nodeOcc, turnOcc, ...
            maxSameDirPerEdge, maxTurnsPerNode, maxNodePassages, T, alt);

        if plan.feasible
            foundPlan = true;
            jobStatus(j)     = "served";
            launchDelay_s(j) = launchTime - tReq;
            flightTime_s(j)  = plan.totalTime;
            routeNodeLists{j}= plan.nodes;
            routeTimes{j}    = plan.times;
            routeDirs{j}     = plan.dirs;
            routeAlts{j}     = plan.alts;
            routeEdgeRows{j} = plan.edgeRows;
            turnCounts(j)    = plan.nTurns;

            % Reserve the route in the tables
            [edgeOcc,nodeOcc,turnOcc] = reservePlan(plan, edgeOcc, nodeOcc, turnOcc, T);
            break;
        end
    end

    if ~foundPlan
        jobStatus(j) = "dropped";
    end
end

%% ===================== 7. POST-SIM METRICS =============================

servedMask  = jobStatus == "served";
droppedMask = jobStatus == "dropped";

servedIdx = find(servedMask);
servedCount = nnz(servedMask);
droppedCount = nnz(droppedMask);

manhattanFt = nan(nJobs,1);
euclidFt    = nan(nJobs,1);
detourRatio = nan(nJobs,1);

for ii = 1:servedCount
    j = servedIdx(ii);
    nodes = routeNodeLists{j};
    x = nodeX(nodes); y = nodeY(nodes);
    manhattanFt(j) = sum(abs(diff(x)) + abs(diff(y)));
    euclidFt(j)    = hypot(x(end)-x(1), y(end)-y(1));
    if euclidFt(j) > 0
        detourRatio(j) = manhattanFt(j) / euclidFt(j);
    end
end

% Basic collision-risk scan using occupied nodes and edges at each time step.
% Since reservations avoid direct simultaneous conflicts, this should mostly be 0.
[riskCount, riskPairs] = scanCollisionRisks(routeNodeLists, routeTimes, nodeX, nodeY, collisionDistFt, servedIdx);

%% ===================== 8. SUMMARY =====================================

fprintf('\n========== SF-STYLE GRID DRONE SIM RESULTS ==========%s', newline);
fprintf('Grid                     : %d x %d nodes (%.0f ft spacing)\n', gridRows, gridCols, gridStepFt);
fprintf('Simulation horizon       : %d s\n', T);
fprintf('Requests generated       : %d\n', nJobs);
fprintf('Requests served          : %d\n', servedCount);
fprintf('Requests dropped         : %d\n', droppedCount);
fprintf('Service rate             : %.1f%%\n', 100*servedCount/max(nJobs,1));
fprintf('Avg launch delay         : %.2f s\n', mean(launchDelay_s(servedMask), 'omitnan'));
fprintf('Avg flight time          : %.2f s\n', mean(flightTime_s(servedMask), 'omitnan'));
fprintf('Avg Manhattan distance   : %.2f ft\n', mean(manhattanFt(servedMask), 'omitnan'));
fprintf('Avg Euclidean distance   : %.2f ft\n', mean(euclidFt(servedMask), 'omitnan'));
fprintf('Avg detour ratio         : %.3f\n', mean(detourRatio(servedMask), 'omitnan'));
fprintf('Avg turns per flight     : %.2f\n', mean(turnCounts(servedMask), 'omitnan'));
fprintf('Collision-risk pairs     : %d\n', riskCount);
fprintf('Blocked building nodes   : %d\n', nnz(blockedNode));
fprintf('Altitude bands [ft]      : N=%d, S=%d, E=%d, W=%d, turn=%d\n', ...
    alt.N, alt.S, alt.E, alt.W, alt.turn);

%% ===================== 9. VISUALIZATIONS ===============================

figure('Position',[50 50 1600 900],'Name','SF Grid Drone Delivery Simulation');

% -------- 9a. Top-down map --------
subplot(2,3,1); hold on; axis equal; grid on;
title('Top-Down Routes + Building Exclusions');
xlabel('x [ft]'); ylabel('y [ft]');

% grid lines
for r = 1:gridRows
    yy = nodeY(nodeID(r,1));
    plot([0,(gridCols-1)*gridStepFt],[yy,yy],'-','Color',[0.88 0.88 0.88]);
end
for c = 1:gridCols
    xx = nodeX(nodeID(1,c));
    plot([xx,xx],[0,(gridRows-1)*gridStepFt],'-','Color',[0.88 0.88 0.88]);
end

% blocked nodes/building zones
[br,bc] = find(blockedNode);
if ~isempty(br)
    scatter((bc-1)*gridStepFt, (gridRows-br)*gridStepFt, 35, buildingHeightFt(blockedNode), 's', 'filled');
    colormap(gca, parula);
    cb = colorbar; ylabel(cb,'Building height [ft]');
end

plot(nodeX(depotNodeIDs), nodeY(depotNodeIDs), 'kp', 'MarkerSize', 12, 'MarkerFaceColor', 'y');

nPlot = min(servedCount, 40);
if nPlot > 0
    cmap = lines(nPlot);
    for ii = 1:nPlot
        j = servedIdx(ii);
        nodes = routeNodeLists{j};
        plot(nodeX(nodes), nodeY(nodes), '-', 'Color', [cmap(ii,:) 0.55], 'LineWidth', 1.2);
        plot(nodeX(nodes(1)), nodeY(nodes(1)), 'go', 'MarkerFaceColor', 'g', 'MarkerSize', 4);
        plot(nodeX(nodes(end)), nodeY(nodes(end)), 'rs', 'MarkerFaceColor', 'r', 'MarkerSize', 4);
    end
end
legend({'','','Blocked / tall building nodes','Depots','Routes','Origin','Destination'}, 'Location','bestoutside');

% -------- 9b. 3D trajectories --------
subplot(2,3,2); hold on; grid on; view(35,25);
title('3D Trajectories (First 8 Served Jobs)');
xlabel('x [ft]'); ylabel('y [ft]'); zlabel('Altitude [ft]');

% show buildings as vertical bars
[br,bc] = find(buildingHeightFt > 0);
for kk = 1:numel(br)
    x0 = (bc(kk)-1)*gridStepFt;
    y0 = (gridRows-br(kk))*gridStepFt;
    h  = buildingHeightFt(br(kk),bc(kk));
    plot3([x0 x0],[y0 y0],[0 h],'-','Color',[0.5 0.5 0.5],'LineWidth',2);
end

for ii = 1:min(servedCount,8)
    j = servedIdx(ii);
    nodes = routeNodeLists{j};
    dirs  = routeDirs{j};
    alts  = routeAlts{j};
    [wx,wy,wz] = build3DTrajectory(nodes, dirs, alts, nodeX, nodeY, alt);
    plot3(wx,wy,wz,'-','LineWidth',1.5);
end
zlim([0 maxAltitudeFt]);

% -------- 9c. Demand over time --------
subplot(2,3,3); hold on; grid on;
title('Demand Time Series');
plot(timeVec, lambda_t, 'b-', 'LineWidth', 1.8);
bar(timeVec, arrivalsPerStep, 1, 'FaceAlpha', 0.25, 'EdgeAlpha', 0.15);
xlabel('Time [s]'); ylabel('Requests / s');
legend('Demand rate \lambda(t)','Realized arrivals');

% -------- 9d. Flight time distribution --------
subplot(2,3,4);
histogram(flightTime_s(servedMask), 20, 'FaceColor', [0.90 0.40 0.25], 'FaceAlpha', 0.8);
grid on; xlabel('Flight time [s]'); ylabel('Count');
title('Flight Time Distribution');
xline(mean(flightTime_s(servedMask),'omitnan'),'b--','LineWidth',1.5);

% -------- 9e. Delay vs request time --------
subplot(2,3,5); hold on; grid on;
servedReqTimes = arrayfun(@(s) s.requestTime, jobs(servedMask));
scatter(servedReqTimes, launchDelay_s(servedMask), 18, turnCounts(servedMask), 'filled');
cb2 = colorbar; ylabel(cb2,'Turns per trip');
xlabel('Request time [s]'); ylabel('Launch delay [s]');
title('Delay vs Request Time');

% -------- 9f. Distance comparison --------
subplot(2,3,6); hold on; grid on;
scatter(euclidFt(servedMask), manhattanFt(servedMask), 18, launchDelay_s(servedMask), 'filled');
mx = max(manhattanFt(servedMask),[],'omitnan');
plot([0 mx],[0 mx],'k--','LineWidth',1);
colorbar; xlabel('Euclidean [ft]'); ylabel('Routed Manhattan [ft]');
title('Route Penalty from Grid Constraint');

sgtitle('Decentralized Drone Delivery on a Building-Constrained SF Grid','FontWeight','bold');

%% ===================== 10. EXTRA VISUALS ===============================

% A. Service heatmap at destinations
figure('Position',[100 100 1200 450],'Name','Extra Visualizations');
subplot(1,3,1);
servedDestCounts = zeros(gridRows, gridCols);
for ii = 1:servedCount
    j = servedIdx(ii);
    [r,c] = ind2sub([gridRows,gridCols], jobs(j).dest);
    servedDestCounts(r,c) = servedDestCounts(r,c) + 1;
end
imagesc(servedDestCounts); axis image; colorbar;
title('Served Destination Heatmap'); xlabel('Col'); ylabel('Row');

% B. Launch delay histogram
subplot(1,3,2);
histogram(launchDelay_s(servedMask), 20, 'FaceColor', [0.2 0.5 0.9], 'FaceAlpha', 0.85);
grid on; xlabel('Launch delay [s]'); ylabel('Count'); title('Launch Delay Distribution');

% C. Edge utilization map
subplot(1,3,3); hold on; axis equal; grid on;
title('Edge Utilization Map'); xlabel('x [ft]'); ylabel('y [ft]');
edgeUseCount = zeros(size(edgeList,1),1);
for j = servedIdx'
    er = routeEdgeRows{j};
    edgeUseCount(er) = edgeUseCount(er) + 1;
end
maxUse = max(edgeUseCount);
for e = 1:size(edgeList,1)
    if edgeUseCount(e) == 0, continue; end
    u = edgeList(e,1); v = edgeList(e,2);
    lw = 0.5 + 4*edgeUseCount(e)/maxUse;
    plot([nodeX(u) nodeX(v)], [nodeY(u) nodeY(v)], '-', 'LineWidth', lw);
end
plot(nodeX(depotNodeIDs), nodeY(depotNodeIDs), 'kp', 'MarkerFaceColor','y', 'MarkerSize',10);

fprintf('\nSimulation complete.\n');

%% ===================== LOCAL FUNCTIONS =================================

function d = manhattanNodeDistance(origins, dest, nRows, nCols)
    [ro,co] = ind2sub([nRows,nCols], origins);
    [rd,cd] = ind2sub([nRows,nCols], dest);
    d = abs(ro-rd) + abs(co-cd);
end

function plan = planRouteTimeAware(origin, dest, launchTime, edgeList, blockedEdge, ...
    blockedNode, nRows, nCols, speedFt_s, stepFt, turnPenalty_s, takeoff_s, landing_s, ...
    edgeOcc, nodeOcc, turnOcc, maxSameDirPerEdge, maxTurnsPerNode, maxNodePassages, T, alt)

    % Time-expanded Dijkstra / label-setting method.
    % State = (node, time, prevDir)
    % prevDir in {0,1,2,3,4} meaning none,N,S,E,W

    segTravel = ceil(stepFt / speedFt_s);   % integer seconds per edge
    nNodes = nRows*nCols;
    maxTime = T;
    INF = 1e12;

    % prevDir map
    DIR_NONE = 0; DIR_N = 1; DIR_S = 2; DIR_E = 3; DIR_W = 4;

    dist = INF * ones(nNodes, maxTime+1, 5);
    prevNode = zeros(nNodes, maxTime+1, 5, 'uint32');
    prevTime = zeros(nNodes, maxTime+1, 5, 'uint32');
    prevDirA = zeros(nNodes, maxTime+1, 5, 'uint8');

    t0 = launchTime + takeoff_s;
    if t0 > maxTime
        plan = infeasiblePlan(); return;
    end

    dist(origin, t0+1, DIR_NONE+1) = t0;
    open = [origin, t0, DIR_NONE];

    found = false;
    bestFinal = [];

    while ~isempty(open)
        % Extract state with smallest current time cost
        costs = arrayfun(@(k) dist(open(k,1), open(k,2)+1, open(k,3)+1), 1:size(open,1));
        [~,idx] = min(costs);
        state = open(idx,:);
        open(idx,:) = [];

        u = state(1); t = state(2); prevDir = state(3);
        curCost = dist(u, t+1, prevDir+1);

        if u == dest
            found = true;
            bestFinal = [u,t,prevDir];
            break;
        end

        % Outgoing neighbors in edge list
        outRows = find(edgeList(:,1) == u & ~blockedEdge);
        for rr = outRows'
            v = edgeList(rr,2);
            [ru,cu] = ind2sub([nRows,nCols],u);
            [rv,cv] = ind2sub([nRows,nCols],v);
            if blockedNode(rv,cv), continue; end

            curDir = getDirection(ru,cu,rv,cv, DIR_N, DIR_S, DIR_E, DIR_W);
            turnCost = 0;
            doTurn = false;
            if prevDir ~= DIR_NONE && prevDir ~= curDir
                turnCost = turnPenalty_s;
                doTurn = true;
            end

            tEnter = t + turnCost;
            tExit  = tEnter + segTravel;
            if tExit > maxTime, continue; end

            % Check reservations during turn and travel
            if doTurn
                if ~nodeTurnAvailable(u, t+1, tEnter, turnOcc, maxTurnsPerNode)
                    continue;
                end
            end

            if ~edgeAvailable(rr, tEnter+1, tExit, edgeOcc, maxSameDirPerEdge)
                continue;
            end
            if ~nodeAvailable(v, tExit, nodeOcc, maxNodePassages)
                continue;
            end

            newCost = curCost + turnCost + segTravel;
            if newCost < dist(v, tExit+1, curDir+1)
                dist(v, tExit+1, curDir+1) = newCost;
                prevNode(v, tExit+1, curDir+1) = u;
                prevTime(v, tExit+1, curDir+1) = t;
                prevDirA(v, tExit+1, curDir+1) = prevDir;
                open(end+1,:) = [v, tExit, curDir]; %#ok<AGROW>
            end
        end
    end

    if ~found
        plan = infeasiblePlan(); return;
    end

    % Reconstruct
    nodes = bestFinal(1);
    times = bestFinal(2);
    dirs  = [];
    dirNow = bestFinal(3);
    nodeNow = bestFinal(1);
    timeNow = bestFinal(2);

    while ~(nodeNow == origin && timeNow == t0 && dirNow == DIR_NONE)
        pNode = double(prevNode(nodeNow, timeNow+1, dirNow+1));
        pTime = double(prevTime(nodeNow, timeNow+1, dirNow+1));
        pDir  = double(prevDirA(nodeNow, timeNow+1, dirNow+1));

        nodes = [pNode nodes]; %#ok<AGROW>
        times = [pTime times]; %#ok<AGROW>
        dirs  = [dirNow dirs]; %#ok<AGROW>

        nodeNow = pNode;
        timeNow = pTime;
        dirNow  = pDir;
    end

    nSeg = numel(nodes)-1;
    alts = zeros(1,nSeg);
    edgeRows = zeros(1,nSeg);
    nTurns = 0;
    for s = 1:nSeg
        [r1,c1] = ind2sub([nRows,nCols], nodes(s));
        [r2,c2] = ind2sub([nRows,nCols], nodes(s+1));
        edgeRows(s) = find(edgeList(:,1)==nodes(s) & edgeList(:,2)==nodes(s+1), 1, 'first');
        switch dirs(s)
            case DIR_N, alts(s) = alt.N;
            case DIR_S, alts(s) = alt.S;
            case DIR_E, alts(s) = alt.E;
            case DIR_W, alts(s) = alt.W;
        end
        if s > 1 && dirs(s) ~= dirs(s-1)
            nTurns = nTurns + 1;
        end
    end

    plan.feasible = true;
    plan.nodes = nodes;
    plan.times = times;
    plan.dirs  = dirs;
    plan.alts  = alts;
    plan.edgeRows = edgeRows;
    plan.nTurns = nTurns;
    plan.totalTime = (times(end) - launchTime) + landing_s;
end

function ok = edgeAvailable(edgeRow, t1, t2, edgeOcc, cap)
    ok = true;
    for tt = t1:min(t2, numel(edgeOcc))
        if isempty(edgeOcc{tt}), continue; end
        ok = nnz(edgeOcc{tt} == edgeRow) < cap;
        if ~ok, return; end
    end
end

function ok = nodeAvailable(node, t, nodeOcc, cap)
    if t+1 > numel(nodeOcc), ok = false; return; end
    if isempty(nodeOcc{t+1})
        ok = true;
    else
        ok = nnz(nodeOcc{t+1} == node) < cap;
    end
end

function ok = nodeTurnAvailable(node, t1, t2, turnOcc, cap)
    ok = true;
    for tt = t1:min(t2, numel(turnOcc)-1)
        if isempty(turnOcc{tt}), continue; end
        ok = nnz(turnOcc{tt} == node) < cap;
        if ~ok, return; end
    end
end

function [edgeOcc,nodeOcc,turnOcc] = reservePlan(plan, edgeOcc, nodeOcc, turnOcc, T)
    nodes = plan.nodes;
    times = plan.times;
    dirs  = plan.dirs;
    edgeRows = plan.edgeRows;

    for s = 1:numel(edgeRows)
        tStart = times(s);
        tEnd   = times(s+1);

        if s > 1 && dirs(s) ~= dirs(s-1)
            for tt = tStart+1:min(tStart+2, T)
                turnOcc{tt+1}(end+1) = nodes(s); %#ok<AGROW>
            end
        end

        for tt = tStart+1:min(tEnd, T)
            edgeOcc{tt+1}(end+1) = edgeRows(s); %#ok<AGROW>
        end
        nodeOcc{tEnd+1}(end+1) = nodes(s+1); %#ok<AGROW>
    end
end

function dir = getDirection(r1,c1,r2,c2, DIR_N, DIR_S, DIR_E, DIR_W)
    if r2 < r1
        dir = DIR_N;
    elseif r2 > r1
        dir = DIR_S;
    elseif c2 > c1
        dir = DIR_E;
    else
        dir = DIR_W;
    end
end

function plan = infeasiblePlan()
    plan = struct('feasible', false, 'nodes', [], 'times', [], 'dirs', [], ...
        'alts', [], 'edgeRows', [], 'nTurns', [], 'totalTime', []);
end

function [wx,wy,wz] = build3DTrajectory(nodes, dirs, alts, nodeX, nodeY, alt)
    wx = nodeX(nodes(1));
    wy = nodeY(nodes(1));
    wz = 0;

    if ~isempty(alts)
        wx(end+1) = nodeX(nodes(1));
        wy(end+1) = nodeY(nodes(1));
        wz(end+1) = alts(1);
    end

    for s = 1:numel(dirs)
        if s > 1 && dirs(s) ~= dirs(s-1)
            wx(end+1) = nodeX(nodes(s)); %#ok<AGROW>
            wy(end+1) = nodeY(nodes(s)); %#ok<AGROW>
            wz(end+1) = alt.turn; %#ok<AGROW>
            wx(end+1) = nodeX(nodes(s)); %#ok<AGROW>
            wy(end+1) = nodeY(nodes(s)); %#ok<AGROW>
            wz(end+1) = alts(s); %#ok<AGROW>
        end
        wx(end+1) = nodeX(nodes(s+1)); %#ok<AGROW>
        wy(end+1) = nodeY(nodes(s+1)); %#ok<AGROW>
        wz(end+1) = alts(s); %#ok<AGROW>
    end

    wx(end+1) = nodeX(nodes(end));
    wy(end+1) = nodeY(nodes(end));
    wz(end+1) = 0;
end

function [riskCount, riskPairs] = scanCollisionRisks(routeNodeLists, routeTimes, nodeX, nodeY, collisionDistFt, servedIdx)
    riskCount = 0;
    riskPairs = [];
    for a = 1:numel(servedIdx)
        ja = servedIdx(a);
        nodesA = routeNodeLists{ja};
        timesA = routeTimes{ja};
        if isempty(nodesA), continue; end
        for b = a+1:numel(servedIdx)
            jb = servedIdx(b);
            nodesB = routeNodeLists{jb};
            timesB = routeTimes{jb};
            if isempty(nodesB), continue; end

            commonT = intersect(timesA, timesB);
            flagged = false;
            for tt = commonT
                ia = find(timesA == tt, 1, 'first');
                ib = find(timesB == tt, 1, 'first');
                d = hypot(nodeX(nodesA(ia)) - nodeX(nodesB(ib)), nodeY(nodesA(ia)) - nodeY(nodesB(ib)));
                if d < collisionDistFt
                    riskCount = riskCount + 1;
                    riskPairs = [riskPairs; ja jb tt d]; %#ok<AGROW>
                    flagged = true;
                    break;
                end
            end
            if flagged
                continue;
            end
        end
    end
end
function r = my_poissrnd(lambda)
% Generate a Poisson random number without toolboxes
L = exp(-lambda);
k = 0;
p = 1;

while p > L
    k = k + 1;
    p = p * rand();
end

r = k - 1;
end
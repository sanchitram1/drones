%% ========================================================================
%  DECENTRALIZED DRONE DELIVERY AIRSPACE SIMULATION
%  IEOR 290 / Transportation Analytics -- Spring 2026
%  ------------------------------------------------------------------------
%  Single-file MATLAB script.  Runs top-to-bottom in one Jupyter cell
%  using the MATLAB kernel, or as  `run('drone_airspace_sim.m')`.
%
%  Research question:
%    "At what drone density does each topology's conflict rate become
%     unacceptable, and what is the throughput-energy tradeoff?"
%
%  Two topologies compared:
%    (a) Manhattan grid              -- 4 altitude bands (N/S/E/W)
%    (b) Grid + diagonal overlay     -- 8 altitude bands (+ NE/SW/NW/SE)
%
%  Trip structure (three-leg): depot -> restaurant -> residence
%
%  Turning protocol: "intersection cube" -- drones ascend to a transition
%  altitude above the intersection, rotate heading, then descend to the
%  band associated with the new bearing.  This guarantees that two lanes
%  never occupy the same (x,y,z) cell simultaneously.
%
%  Optional data files (placed in the same working directory):
%    * sf_restaurants.csv        cols: name, lat, lon          (Kaggle Uber Eats)
%    * sf_census_tracts.csv      cols: tract_id, lat, lon, pop (DataSF + Census)
%  If absent, the script generates realistic synthetic equivalents so the
%  simulation runs anywhere.
%
%  Toolbox requirements:
%    * Core MATLAB          (graph / shortestpath)
%    * Statistics Toolbox   (kmeans -- depot placement)
%  Optional (for nicer plots): Mapping Toolbox.  NOT required.
%  ========================================================================

clear; clc; close all;
rng('default');

%% ------------------------------------------------------------------------
%% 0.  GLOBAL CONFIGURATION
%% ------------------------------------------------------------------------
cfg = struct();

% ---- Study area: SF Mission / SoMa / Downtown core ---------------------
cfg.area.latRange   = [37.7550 37.7950];
cfg.area.lonRange   = [-122.4200 -122.3900];
cfg.area.refLat     = mean(cfg.area.latRange);
cfg.area.refLon     = mean(cfg.area.lonRange);
cfg.area.metersPerDegLat = 111320;
cfg.area.metersPerDegLon = 111320 * cosd(cfg.area.refLat);

% ---- Grid topology (synthetic Manhattan grid approximating SF) ---------
cfg.grid.nRows      = 20;      % intersections N-S  (~4.4 km span)
cfg.grid.nCols      = 20;      % intersections E-W  (~2.6 km span)
cfg.grid.spacingM   = 220;     % ~1 SF city block ≈ 220-260 m

% ---- Altitude bands (meters AGL) --------------------------------------
% 4-band grid topology: N/S/E/W
% 8-band overlay:       N/S/E/W + NE/SW/NW/SE
cfg.alt.N  = 50;  cfg.alt.S  = 60;   cfg.alt.E  = 70;   cfg.alt.W  = 80;
cfg.alt.NE = 90;  cfg.alt.SW = 100;  cfg.alt.NW = 110;  cfg.alt.SE = 120;
cfg.alt.transitionCube = 135;        % ceiling inside intersection cube
cfg.alt.buffer         = 10;         % vertical separation threshold (m)

% ---- Drone performance -------------------------------------------------
cfg.drone.cruiseSpeed  = 15;    % m/s  (typical delivery quadcopter)
cfg.drone.climbSpeed   = 3;     % m/s
cfg.drone.hoverPower   = 300;   % W
cfg.drone.cruisePower  = 220;   % W
cfg.drone.pickupTime   = 45;    % s  (restaurant dwell)
cfg.drone.deliveryTime = 30;    % s  (residence dwell)

% ---- Demand & depots ---------------------------------------------------
cfg.depots.nDepots        = 5;           % k-means on restaurants
cfg.demand.nRestaurants   = 120;         % synthetic if no Kaggle file
cfg.demand.nResidences    = 400;         % synthetic if no census file
cfg.demand.simDurationS   = 1800;        % 30-minute horizon
cfg.demand.arrivalLambda  = @(n) n/60;   % 1 order per drone per 60 s

% ---- Conflict detection ------------------------------------------------
cfg.conflict.hThresholdM  = 50;    % horizontal separation standard
cfg.conflict.vThresholdM  = 10;    % vertical separation standard
cfg.conflict.dtS          = 2;     % simulation tick (seconds)

% ---- Density sweep -----------------------------------------------------
cfg.sweep.droneCounts     = [20 40 60 80 120 160 220];
cfg.sweep.nSeeds          = 3;
cfg.sweep.topologies      = ["grid","diagonal"];

% ---- Output ------------------------------------------------------------
cfg.output.saveMat        = true;
cfg.output.savePng        = true;
cfg.output.outDir         = pwd;

fprintf('=== DECENTRALIZED DRONE AIRSPACE SIMULATION ===\n');
fprintf('Study area: SF [%.4f,%.4f] x [%.4f,%.4f]\n', ...
    cfg.area.latRange(1), cfg.area.latRange(2), ...
    cfg.area.lonRange(1), cfg.area.lonRange(2));
fprintf('Grid: %dx%d intersections, %d m spacing\n', ...
    cfg.grid.nRows, cfg.grid.nCols, cfg.grid.spacingM);
fprintf('Sweep: %d densities x %d seeds x %d topologies = %d runs\n\n', ...
    numel(cfg.sweep.droneCounts), cfg.sweep.nSeeds, ...
    numel(cfg.sweep.topologies), ...
    numel(cfg.sweep.droneCounts)*cfg.sweep.nSeeds*numel(cfg.sweep.topologies));

%% ------------------------------------------------------------------------
%% 1.  BUILD THE STREET-GRID GRAPH (SYNTHETIC MANHATTAN GRID)
%% ------------------------------------------------------------------------
fprintf('[1/8] Building street-grid graph...\n');

network = buildGridNetwork(cfg);

fprintf('     nodes: %d, edges: %d, span: %.2f km N-S x %.2f km E-W\n', ...
    numnodes(network.G), numedges(network.G), ...
    cfg.grid.nRows*cfg.grid.spacingM/1000, ...
    cfg.grid.nCols*cfg.grid.spacingM/1000);

%% ------------------------------------------------------------------------
%% 2.  LOAD OR GENERATE RESTAURANT LOCATIONS  (Kaggle Uber Eats fallback)
%% ------------------------------------------------------------------------
fprintf('[2/8] Loading restaurants...\n');

restaurantFile = fullfile(cfg.output.outDir,'sf_restaurants.csv');
if isfile(restaurantFile)
    T = readtable(restaurantFile);
    mask = T.lat >= cfg.area.latRange(1) & T.lat <= cfg.area.latRange(2) & ...
           T.lon >= cfg.area.lonRange(1) & T.lon <= cfg.area.lonRange(2);
    T = T(mask,:);
    [rx, ry] = latlonToLocalM(T.lat, T.lon, cfg.area);
    restaurants = struct('x',rx,'y',ry,'name',{T.name});
    fprintf('     Loaded %d restaurants from %s\n', numel(rx), restaurantFile);
else
    restaurants = synthesizeRestaurants(cfg, network);
    fprintf('     Synthesized %d restaurants clustered along corridors\n', ...
        numel(restaurants.x));
end

%% ------------------------------------------------------------------------
%% 3.  PLACE DEPOTS VIA K-MEANS ON RESTAURANT COORDS
%% ------------------------------------------------------------------------
fprintf('[3/8] Placing depots via k-means (k=%d)...\n', cfg.depots.nDepots);

restCoords = [restaurants.x, restaurants.y];
[~, depotCentroids] = kmeans(restCoords, cfg.depots.nDepots, ...
    'Replicates', 10, 'Start', 'plus', 'Display', 'off');

% Snap each depot to its nearest intersection node
depots = struct();
depots.x = zeros(cfg.depots.nDepots,1);
depots.y = zeros(cfg.depots.nDepots,1);
depots.node = zeros(cfg.depots.nDepots,1);
for k = 1:cfg.depots.nDepots
    d2 = (network.nodeX - depotCentroids(k,1)).^2 + ...
         (network.nodeY - depotCentroids(k,2)).^2;
    [~, idx] = min(d2);
    depots.node(k) = idx;
    depots.x(k)    = network.nodeX(idx);
    depots.y(k)    = network.nodeY(idx);
end
fprintf('     Depots snapped to nearest grid intersections\n');

%% ------------------------------------------------------------------------
%% 4.  GENERATE RESIDENCES (census-weighted synthetic)
%% ------------------------------------------------------------------------
fprintf('[4/8] Generating residential delivery destinations...\n');

censusFile = fullfile(cfg.output.outDir,'sf_census_tracts.csv');
if isfile(censusFile)
    C = readtable(censusFile);
    mask = C.lat >= cfg.area.latRange(1) & C.lat <= cfg.area.latRange(2) & ...
           C.lon >= cfg.area.lonRange(1) & C.lon <= cfg.area.lonRange(2);
    C = C(mask,:);
    [cx, cy] = latlonToLocalM(C.lat, C.lon, cfg.area);
    w = C.pop / sum(C.pop);
    residences = sampleResidences(cx, cy, w, cfg.demand.nResidences, 80);
    fprintf('     Sampled %d residences from %d census tracts\n', ...
        cfg.demand.nResidences, numel(cx));
else
    residences = synthesizeResidences(cfg, network);
    fprintf('     Synthesized %d residences (log-normal density field)\n', ...
        numel(residences.x));
end

%% ------------------------------------------------------------------------
%% 5.  PRE-COMPUTE PARAMETERS FOR BOTH TOPOLOGIES
%% ------------------------------------------------------------------------
fprintf('[5/8] Preparing topologies...\n');

topoParams = struct();
topoParams.grid     = makeTopology('grid',     cfg);
topoParams.diagonal = makeTopology('diagonal', cfg);

fprintf('     grid     : %d directional bands at altitudes [%s] m\n', ...
    topoParams.grid.nBands,     num2str(topoParams.grid.altitudes));
fprintf('     diagonal : %d directional bands at altitudes [%s] m\n', ...
    topoParams.diagonal.nBands, num2str(topoParams.diagonal.altitudes));

%% ------------------------------------------------------------------------
%% 6.  DENSITY SWEEP
%% ------------------------------------------------------------------------
fprintf('[6/8] Running density sweep...\n');

nDen   = numel(cfg.sweep.droneCounts);
nSeeds = cfg.sweep.nSeeds;
topos  = cfg.sweep.topologies;

totalRuns = nDen * nSeeds * numel(topos);

% Pre-allocate results as a struct with array fields; convert to table
% at the end.  (Using struct() with array args builds a struct array;
% field-by-field assignment is required for scalar struct with arrays.)
results = struct();
results.topology        = repmat("", totalRuns, 1);
results.nDrones         = zeros(totalRuns, 1);
results.seed            = zeros(totalRuns, 1);
results.nMissions       = zeros(totalRuns, 1);
results.nConflicts      = zeros(totalRuns, 1);
results.conflictsPer1k  = zeros(totalRuns, 1);
results.avgFlightTimeS  = zeros(totalRuns, 1);
results.avgDetourRatio  = zeros(totalRuns, 1);
results.avgEnergyWh     = zeros(totalRuns, 1);
results.throughputPerHr = zeros(totalRuns, 1);

row = 0;
tRun0 = tic;

for iTopo = 1:numel(topos)
    topoName = topos(iTopo);
    tp = topoParams.(topoName);
    for iD = 1:nDen
        nDrones = cfg.sweep.droneCounts(iD);
        for iS = 1:nSeeds
            row = row + 1;
            seed = 1000*iD + iS;
            rng(seed);

            % --- generate missions for this run --------------------
            missions = generateMissions(cfg, network, depots, ...
                restaurants, residences, nDrones);

            % --- plan + simulate -----------------------------------
            simOut = simulateFleet(cfg, network, tp, missions);

            % --- record --------------------------------------------
            results.topology(row)        = topoName;
            results.nDrones(row)         = nDrones;
            results.seed(row)            = seed;
            results.nMissions(row)       = numel(missions);
            results.nConflicts(row)      = simOut.nConflicts;
            results.conflictsPer1k(row)  = 1000*simOut.nConflicts / max(1,numel(missions));
            results.avgFlightTimeS(row)  = simOut.avgFlightTimeS;
            results.avgDetourRatio(row)  = simOut.avgDetourRatio;
            results.avgEnergyWh(row)     = simOut.avgEnergyWh;
            results.throughputPerHr(row) = simOut.throughputPerHr;

            fprintf(['     [%3d/%3d]  %-9s n=%3d seed=%d  ' ...
                     'conf=%3d  ft=%.0fs  det=%.2f  thr=%.0f/hr\n'], ...
                row, totalRuns, char(topoName), nDrones, seed, ...
                simOut.nConflicts, simOut.avgFlightTimeS, ...
                simOut.avgDetourRatio, simOut.throughputPerHr);
        end
    end
end

% Convert accumulated struct to table for downstream grouping/plotting.
results = struct2table(results);

fprintf('     Sweep complete in %.1f s\n', toc(tRun0));

%% ------------------------------------------------------------------------
%% 7.  AGGREGATE + PLOT
%% ------------------------------------------------------------------------
fprintf('[7/8] Aggregating and plotting...\n');

agg = groupsummary(results, {'topology','nDrones'}, ...
    {'mean','std'}, ...
    {'nConflicts','conflictsPer1k','avgFlightTimeS', ...
     'avgDetourRatio','avgEnergyWh','throughputPerHr'});

fig = makeResultsFigure(agg, topoParams, network, depots, ...
    restaurants, residences, cfg);

if cfg.output.savePng
    pngPath = fullfile(cfg.output.outDir,'drone_sim_results.png');
    exportgraphics(fig, pngPath, 'Resolution', 150);
    fprintf('     Saved figure to %s\n', pngPath);
end

%% ------------------------------------------------------------------------
%% 8.  IDENTIFY CAPACITY THRESHOLDS + SAVE
%% ------------------------------------------------------------------------
fprintf('[8/8] Identifying capacity thresholds...\n');

threshold = 5;   % capacity == first density where conflictsPer1k > 5
capacity  = struct();
for iTopo = 1:numel(topos)
    topoName = topos(iTopo);
    sub = agg(agg.topology == topoName, :);
    sub = sortrows(sub,'nDrones');
    over = sub.mean_conflictsPer1k > threshold;
    if any(over)
        cap = sub.nDrones(find(over,1,'first'));
    else
        cap = max(sub.nDrones);
    end
    capacity.(topoName) = cap;
    fprintf('     Topology %-9s : capacity (> %d conflicts/1000 flights) = %d drones\n', ...
        topoName, threshold, cap);
end

uplift = 100 * (capacity.diagonal - capacity.grid) / capacity.grid;
fprintf('     Diagonal overlay increases capacity by %+.1f%% vs. plain grid.\n', uplift);

if cfg.output.saveMat
    matPath = fullfile(cfg.output.outDir,'drone_sim_workspace.mat');
    save(matPath,'cfg','network','depots','restaurants','residences', ...
         'topoParams','results','agg','capacity','-v7.3');
    fprintf('     Saved workspace to %s\n', matPath);
end

fprintf('\n=== DONE ===\n');


%% ========================================================================
%%                            LOCAL FUNCTIONS
%% ========================================================================

function network = buildGridNetwork(cfg)
% BUILDGRIDNETWORK   Manhattan-grid street network in local ENU meters.
%   Nodes = intersections; edges = street segments.  Origin = SW corner.
%   Returns a struct with graph G, node coordinates, and edge bearings.

    nR = cfg.grid.nRows;  nC = cfg.grid.nCols;  d = cfg.grid.spacingM;
    nN = nR * nC;

    nodeX = zeros(nN,1);
    nodeY = zeros(nN,1);
    for r = 1:nR
        for c = 1:nC
            k = (r-1)*nC + c;
            nodeX(k) = (c-1)*d;    % x = east
            nodeY(k) = (r-1)*d;    % y = north
        end
    end

    sn = [];  tn = [];  len = [];  bearing = [];  heading = [];
    for r = 1:nR
        for c = 1:nC
            k = (r-1)*nC + c;
            if c < nC
                k2 = (r-1)*nC + (c+1);        % east neighbour
                sn(end+1,1) = k; tn(end+1,1) = k2; %#ok<AGROW>
                len(end+1,1) = d;                                 %#ok<AGROW>
                bearing(end+1,1) = 90;                            %#ok<AGROW>
                heading{end+1,1} = 'E';                           %#ok<AGROW>
            end
            if r < nR
                k2 = r*nC + c;                % north neighbour
                sn(end+1,1) = k; tn(end+1,1) = k2; %#ok<AGROW>
                len(end+1,1) = d;                                 %#ok<AGROW>
                bearing(end+1,1) = 0;                             %#ok<AGROW>
                heading{end+1,1} = 'N';                           %#ok<AGROW>
            end
        end
    end

    EdgeTable = table([sn tn], len, bearing, heading, ...
        'VariableNames', {'EndNodes','Length','Bearing','Heading'});
    NodeTable = table((1:nN)', nodeX, nodeY, ...
        'VariableNames', {'Id','X','Y'});
    G = graph(EdgeTable, NodeTable);

    network.G       = G;
    network.nodeX   = nodeX;
    network.nodeY   = nodeY;
    network.spanX   = (nC-1)*d;
    network.spanY   = (nR-1)*d;
end


function [xM, yM] = latlonToLocalM(lat, lon, area)
    xM = (lon - area.refLon) * area.metersPerDegLon;
    yM = (lat - area.refLat) * area.metersPerDegLat;
    % shift origin to SW corner of the grid area
    xM = xM - (area.lonRange(1) - area.refLon) * area.metersPerDegLon;
    yM = yM - (area.latRange(1) - area.refLat) * area.metersPerDegLat;
end


function restaurants = synthesizeRestaurants(cfg, network)
% Synthetic restaurant field: dense near a "downtown" cluster + arterial
% corridors.  Mimics the SF Mission/SoMa distribution observed in Uber Eats.

    nR = cfg.demand.nRestaurants;
    spanX = network.spanX;  spanY = network.spanY;

    % 70% clustered near downtown centroid
    nClust = round(0.7*nR);
    cx = 0.6*spanX; cy = 0.55*spanY;
    xC = cx + 0.18*spanX*randn(nClust,1);
    yC = cy + 0.18*spanY*randn(nClust,1);

    % 30% along two arterials (a horizontal + a vertical)
    nArt = nR - nClust;
    half = floor(nArt/2);
    xA1 = spanX*rand(half,1);      yA1 = 0.35*spanY + 30*randn(half,1);
    yA2 = spanY*rand(nArt-half,1); xA2 = 0.75*spanX + 30*randn(nArt-half,1);

    xr = [xC; xA1; xA2];
    yr = [yC; yA1; yA2];

    % clip to study area
    mask = xr>=0 & xr<=spanX & yr>=0 & yr<=spanY;
    xr = xr(mask); yr = yr(mask);

    names = arrayfun(@(i) sprintf('Restaurant_%03d',i), (1:numel(xr))', ...
        'UniformOutput', false);

    restaurants.x = xr;
    restaurants.y = yr;
    restaurants.name = names;
end


function residences = synthesizeResidences(cfg, network)
% Residential density field: two population centers with log-normal spread,
% roughly mirroring high-density residential tracts in SoMa/Mission.

    n = cfg.demand.nResidences;
    spanX = network.spanX;  spanY = network.spanY;

    n1 = round(0.6*n); n2 = n - n1;
    % Center 1 (dense)
    x1 = 0.35*spanX + 0.25*spanX*randn(n1,1);
    y1 = 0.65*spanY + 0.25*spanY*randn(n1,1);
    % Center 2 (secondary)
    x2 = 0.75*spanX + 0.20*spanX*randn(n2,1);
    y2 = 0.30*spanY + 0.20*spanY*randn(n2,1);

    xr = [x1; x2];  yr = [y1; y2];
    % clip + keep within bounds
    xr = min(max(xr, 0), spanX);
    yr = min(max(yr, 0), spanY);

    residences.x = xr;
    residences.y = yr;
end


function residences = sampleResidences(cx, cy, w, n, sigmaM)
% Sample residence coordinates from a mixture of Gaussians centered on
% census-tract centroids, weighted by tract population.

    idx = randsample(numel(cx), n, true, w);
    jitter = sigmaM*randn(n,2);
    residences.x = cx(idx) + jitter(:,1);
    residences.y = cy(idx) + jitter(:,2);
end


function tp = makeTopology(kind, cfg)
% MAKETOPOLOGY   Map each compass heading to a cruise altitude.
%   For the grid topology only N/S/E/W are permitted (edges are cardinal).
%   For the diagonal topology the drone may also route along NE/SW/NW/SE
%   virtual links between adjacent intersections.

    tp.kind = kind;
    switch kind
        case 'grid'
            tp.nBands    = 4;
            tp.headings  = {'N','S','E','W'};
            tp.altitudes = [cfg.alt.N, cfg.alt.S, cfg.alt.E, cfg.alt.W];
            tp.allowDiag = false;
        case 'diagonal'
            tp.nBands    = 8;
            tp.headings  = {'N','S','E','W','NE','SW','NW','SE'};
            tp.altitudes = [cfg.alt.N, cfg.alt.S, cfg.alt.E, cfg.alt.W, ...
                            cfg.alt.NE, cfg.alt.SW, cfg.alt.NW, cfg.alt.SE];
            tp.allowDiag = true;
        otherwise
            error('Unknown topology %s', kind);
    end
    tp.altMap = containers.Map(tp.headings, num2cell(tp.altitudes));
    tp.transitionCube = cfg.alt.transitionCube;
end


function missions = generateMissions(cfg, network, depots, restaurants, residences, nDrones)
% Generate a stream of delivery missions (depot -> restaurant -> residence)
% over the simulation horizon.  Each drone performs as many trips as time
% allows.  Arrivals follow a homogeneous Poisson process.

    lambdaPerDrone = cfg.demand.arrivalLambda(nDrones);   % 1/(drone*s)
    T              = cfg.demand.simDurationS;
    totalLambda    = lambdaPerDrone * nDrones;
    expected       = round(totalLambda * T);

    % Generate arrival times
    nArr = max(10, poissrnd(expected));
    arrivals = sort(T * rand(nArr,1));

    nRest = numel(restaurants.x);
    nRes  = numel(residences.x);
    nDep  = cfg.depots.nDepots;

    missions = struct('id',{},'depot',{},'rest',{},'res',{}, ...
        'arrivalT',{},'depotXY',{},'restXY',{},'resXY',{});

    for i = 1:nArr
        d = randi(nDep);
        % restaurant weighted by inverse distance to depot (closer preferred)
        d2 = (restaurants.x - depots.x(d)).^2 + (restaurants.y - depots.y(d)).^2;
        wR = 1 ./ (1 + d2/1e6);
        rIdx = randsample(nRest, 1, true, wR);
        resIdx = randi(nRes);

        missions(i).id       = i;                          %#ok<AGROW>
        missions(i).depot    = d;                          %#ok<AGROW>
        missions(i).rest     = rIdx;                       %#ok<AGROW>
        missions(i).res      = resIdx;                     %#ok<AGROW>
        missions(i).arrivalT = arrivals(i);                %#ok<AGROW>
        missions(i).depotXY  = [depots.x(d), depots.y(d)]; %#ok<AGROW>
        missions(i).restXY   = [restaurants.x(rIdx), restaurants.y(rIdx)]; %#ok<AGROW>
        missions(i).resXY    = [residences.x(resIdx), residences.y(resIdx)]; %#ok<AGROW>
    end
end


function simOut = simulateFleet(cfg, network, tp, missions)
% Plan + simulate all missions.  Returns metrics + conflict count.

    nM = numel(missions);
    if nM == 0
        simOut = struct('nConflicts',0,'avgFlightTimeS',0, ...
            'avgDetourRatio',1,'avgEnergyWh',0,'throughputPerHr',0);
        return
    end

    trajectories = cell(nM,1);
    flightT      = zeros(nM,1);
    detour       = zeros(nM,1);
    energyWh     = zeros(nM,1);

    for i = 1:nM
        m = missions(i);
        traj = planMission(cfg, network, tp, m);
        trajectories{i} = traj;
        flightT(i)  = traj.totalTimeS;
        detour(i)   = traj.detourRatio;
        energyWh(i) = traj.energyWh;
    end

    % --- conflict detection via space-time hashing ------------------------
    nConflicts = countConflicts(trajectories, cfg);

    simOut.nConflicts       = nConflicts;
    simOut.avgFlightTimeS   = mean(flightT);
    simOut.avgDetourRatio   = mean(detour);
    simOut.avgEnergyWh      = mean(energyWh);
    simOut.throughputPerHr  = nM * 3600 / cfg.demand.simDurationS;
end


function traj = planMission(cfg, network, tp, m)
% Plan a single three-leg mission:
%   depot -> (nearest node to restaurant) -> (nearest node to residence)
% Route is shortest-path on the grid graph (Euclidean edge weights).
% Each edge is assigned the altitude band corresponding to its heading.
% At every heading change we insert an intersection-cube climb/descend.

    % Anchor nodes for restaurant + residence
    nRest = nearestNode(network, m.restXY);
    nRes  = nearestNode(network, m.resXY);

    leg1 = shortestpath(network.G, m.depot, nRest);
    leg2 = shortestpath(network.G, nRest,  nRes );

    if isempty(leg1) || isempty(leg2)
        traj = emptyTraj();
        return
    end

    nodeSeq = [leg1, leg2(2:end)];

    % Edge list along the path
    edges = [];  bearings = {}; lengths = [];
    for i = 1:numel(nodeSeq)-1
        nA = nodeSeq(i);   nB = nodeSeq(i+1);
        eIdx = findedge(network.G, nA, nB);
        if eIdx == 0, continue; end
        edges(end+1,1)    = eIdx;                                        %#ok<AGROW>
        lengths(end+1,1)  = network.G.Edges.Length(eIdx);                %#ok<AGROW>
        hdg = network.G.Edges.Heading{eIdx};
        % bearing depends on traversal direction
        if network.G.Edges.EndNodes(eIdx,1) == nA
            bearings{end+1,1} = hdg;                                     %#ok<AGROW>
        else
            bearings{end+1,1} = oppositeHeading(hdg);                    %#ok<AGROW>
        end
    end

    % For the diagonal topology we fold same-bearing consecutive edges into
    % a single synthetic diagonal leg when two orthogonal moves stack.
    if tp.allowDiag
        [edges, bearings, lengths, nodeSeq] = ...
            foldDiagonals(network, nodeSeq, edges, bearings, lengths, tp);
    end

    % Build 4D trajectory (x,y,z,t) with intersection-cube climbs
    traj = assembleTrajectory(cfg, network, tp, nodeSeq, bearings, ...
        lengths, m);
end


function n = nearestNode(network, xy)
    d2 = (network.nodeX - xy(1)).^2 + (network.nodeY - xy(2)).^2;
    [~,n] = min(d2);
end


function h = oppositeHeading(h)
    switch h
        case 'N',  h = 'S';
        case 'S',  h = 'N';
        case 'E',  h = 'W';
        case 'W',  h = 'E';
        case 'NE', h = 'SW';
        case 'SW', h = 'NE';
        case 'NW', h = 'SE';
        case 'SE', h = 'NW';
    end
end


function [edges, bearings, lengths, newSeq] = ...
    foldDiagonals(network, nodeSeq, edges, bearings, lengths, tp)
% If two consecutive legs are orthogonal (e.g. E followed by N) and span
% one block each, fuse them into a NE diagonal leg.  This gives the
% diagonal overlay fewer, longer hops and routes them to the NE/SW/NW/SE
% bands.

    if numel(bearings) < 2, newSeq = nodeSeq; return; end
    i = 1;
    while i < numel(bearings)
        pair = strcat(bearings{i}, bearings{i+1});
        diag = ''; %#ok<NASGU>
        switch pair
            case {'EN','NE'}, diag = 'NE';
            case {'WS','SW'}, diag = 'SW';
            case {'ES','SE'}, diag = 'SE';
            case {'WN','NW'}, diag = 'NW';
            otherwise,        diag = '';
        end
        if ~isempty(diag) && tp.altMap.isKey(diag)
            % merge
            bearings{i} = diag;
            lengths(i)  = hypot(lengths(i), lengths(i+1));
            edges(i)    = -1;              % synthetic diagonal
            bearings(i+1) = [];
            lengths(i+1)  = [];
            edges(i+1)    = [];
            % (do NOT advance i; check next fold)
        else
            i = i + 1;
        end
    end
    newSeq = nodeSeq;   % node sequence kept for reference only
end


function traj = assembleTrajectory(cfg, network, tp, nodeSeq, bearings, lengths, m)
% Walk the route edge-by-edge at cruiseSpeed, inserting vertical climbs
% whenever the bearing changes between consecutive edges.

    dt = cfg.conflict.dtS;
    v  = cfg.drone.cruiseSpeed;
    vc = cfg.drone.climbSpeed;
    t  = m.arrivalT;

    % Current position starts at depot node XY
    curX = network.nodeX(nodeSeq(1));
    curY = network.nodeY(nodeSeq(1));
    curZ = 0;

    times  = t;
    xs     = curX;  ys = curY;  zs = curZ;
    bandS  = {'GROUND'};

    energyWh = 0;

    % Guard: if no edges to traverse (depot == restaurant == residence node)
    if isempty(bearings)
        eucl = hypot(m.restXY(1)-m.depotXY(1), m.restXY(2)-m.depotXY(2)) + ...
               hypot(m.resXY(1)-m.restXY(1),   m.resXY(2)-m.restXY(2));
        hoverT = cfg.drone.pickupTime + cfg.drone.deliveryTime;
        traj.t          = [t; t + hoverT];
        traj.x          = [curX; curX];
        traj.y          = [curY; curY];
        traj.z          = [0; 0];
        traj.band       = {'GROUND';'HOVER'};
        traj.totalTimeS = hoverT;
        traj.pathM      = 0;
        traj.detourRatio = 1 + eucl/max(eucl,1)*0;  % == 1
        traj.energyWh   = cfg.drone.hoverPower * hoverT / 3600;
        traj.startT     = t;
        traj.endT       = t + hoverT;
        return
    end

    % Climb from ground to first band
    firstHdg = bearings{1};
    firstZ   = tp.altMap(firstHdg);
    [times, xs, ys, zs, bandS, t] = addSegment(times,xs,ys,zs,bandS, ...
        curX,curY,curZ, curX,curY,firstZ, vc, dt, t, 'CLIMB');
    energyWh = energyWh + cfg.drone.cruisePower * (firstZ/vc)/3600;
    curZ = firstZ;

    cumDist = 0;
    prevHdg = firstHdg;

    for i = 1:numel(bearings)
        hdg = bearings{i};  L = lengths(i);
        dx = 0; dy = 0;
        switch hdg
            case 'N',  dy = +L;
            case 'S',  dy = -L;
            case 'E',  dx = +L;
            case 'W',  dx = -L;
            case 'NE', dx = +L/sqrt(2); dy = +L/sqrt(2);
            case 'SW', dx = -L/sqrt(2); dy = -L/sqrt(2);
            case 'NW', dx = -L/sqrt(2); dy = +L/sqrt(2);
            case 'SE', dx = +L/sqrt(2); dy = -L/sqrt(2);
        end
        nxtZ = tp.altMap(hdg);

        % If heading changed, do intersection-cube climb to transition
        % altitude, turn, then descend to new band.
        if ~strcmp(hdg, prevHdg)
            [times,xs,ys,zs,bandS,t] = addSegment(times,xs,ys,zs,bandS, ...
                curX,curY,curZ, curX,curY,tp.transitionCube, vc, dt, t, 'TURN_UP');
            [times,xs,ys,zs,bandS,t] = addSegment(times,xs,ys,zs,bandS, ...
                curX,curY,tp.transitionCube, curX,curY,nxtZ, vc, dt, t, 'TURN_DN');
            % energy: climb up from curZ to transitionCube then down to nxtZ
            energyWh = energyWh + cfg.drone.cruisePower * ...
                ((tp.transitionCube - curZ) + (tp.transitionCube - nxtZ))/vc / 3600;
            curZ = nxtZ;
        end

        % cruise segment
        [times,xs,ys,zs,bandS,t] = addSegment(times,xs,ys,zs,bandS, ...
            curX,curY,curZ, curX+dx,curY+dy,nxtZ, v, dt, t, hdg);
        curX = curX + dx;  curY = curY + dy;  curZ = nxtZ;
        cumDist = cumDist + L;
        prevHdg = hdg;
        energyWh = energyWh + cfg.drone.cruisePower * (L/v)/3600;
    end

    % Pickup + delivery hover (approximate: consolidated at end)
    hoverT = cfg.drone.pickupTime + cfg.drone.deliveryTime;
    [times,xs,ys,zs,bandS,t] = addSegment(times,xs,ys,zs,bandS, ...
        curX,curY,curZ, curX,curY,curZ, 1e9, dt, t, 'HOVER', hoverT);
    energyWh = energyWh + cfg.drone.hoverPower*hoverT/3600;

    % Descend at residence
    [times,xs,ys,zs,bandS,t] = addSegment(times,xs,ys,zs,bandS, ...
        curX,curY,curZ, curX,curY,0, vc, dt, t, 'DESCEND');
    energyWh = energyWh + cfg.drone.cruisePower*(curZ/vc)/3600;

    % Euclidean reference distance (depot->rest->res)
    eucl = hypot(m.restXY(1)-m.depotXY(1), m.restXY(2)-m.depotXY(2)) + ...
           hypot(m.resXY(1)-m.restXY(1),   m.resXY(2)-m.restXY(2));

    traj.t          = times(:);
    traj.x          = xs(:);
    traj.y          = ys(:);
    traj.z          = zs(:);
    traj.band       = bandS(:);
    traj.totalTimeS = t - m.arrivalT;
    traj.pathM      = cumDist;
    traj.detourRatio = (cumDist + 0.01) / max(eucl, 1);
    traj.energyWh   = energyWh;
    traj.startT     = m.arrivalT;
    traj.endT       = t;
end


function [tT,xT,yT,zT,bT,tNow] = addSegment(tT,xT,yT,zT,bT, ...
        x0,y0,z0, x1,y1,z1, v, dt, t0, label, dwellOverride)
% Append a straight-line segment sampled at dt seconds.
    if nargin < 15, dwellOverride = []; end
    dx = x1-x0; dy = y1-y0; dz = z1-z0;
    L  = sqrt(dx^2 + dy^2 + dz^2);
    if ~isempty(dwellOverride)
        duration = dwellOverride;
    else
        duration = L / v;
    end
    if duration < dt
        tT(end+1,1) = t0 + duration; %#ok<AGROW>
        xT(end+1,1) = x1;             %#ok<AGROW>
        yT(end+1,1) = y1;             %#ok<AGROW>
        zT(end+1,1) = z1;             %#ok<AGROW>
        bT{end+1,1} = label;          %#ok<AGROW>
        tNow = t0 + duration;
        return
    end
    nStep = max(1, round(duration/dt));
    ts    = linspace(dt, duration, nStep)';
    frac  = ts / duration;
    xs    = x0 + frac*dx;
    ys    = y0 + frac*dy;
    zs    = z0 + frac*dz;
    tT = [tT; t0 + ts];
    xT = [xT; xs];
    yT = [yT; ys];
    zT = [zT; zs];
    bT = [bT; repmat({label}, nStep, 1)];
    tNow = t0 + duration;
end


function nConflicts = countConflicts(trajectories, cfg)
% Space-time hashed pairwise conflict detection.
% Two drones are in conflict if at the SAME simulation tick their
% horizontal distance < hThresholdM AND their vertical distance < vThresholdM.
% We hash (tick, cellX, cellY) and check pairs inside each bucket.

    cellSize = cfg.conflict.hThresholdM;
    dt       = cfg.conflict.dtS;
    vThr     = cfg.conflict.vThresholdM;

    % Resample every trajectory onto the global tick grid
    allT = [];
    allX = [];
    allY = [];
    allZ = [];
    allID = [];
    for i = 1:numel(trajectories)
        tr = trajectories{i};
        if isempty(tr.t), continue; end
        t0 = tr.t(1); t1 = tr.t(end);
        ts = (ceil(t0/dt)*dt : dt : floor(t1/dt)*dt)';
        if isempty(ts), continue; end
        xs = interp1(tr.t, tr.x, ts, 'linear');
        ys = interp1(tr.t, tr.y, ts, 'linear');
        zs = interp1(tr.t, tr.z, ts, 'linear');
        ids = i*ones(numel(ts),1);
        allT  = [allT;  ts]; %#ok<AGROW>
        allX  = [allX;  xs]; %#ok<AGROW>
        allY  = [allY;  ys]; %#ok<AGROW>
        allZ  = [allZ;  zs]; %#ok<AGROW>
        allID = [allID; ids];%#ok<AGROW>
    end

    if isempty(allT), nConflicts = 0; return; end

    tick  = round(allT/dt);
    cellX = floor(allX/cellSize);
    cellY = floor(allY/cellSize);

    % Only count drones in the SAME cell at the SAME tick (cheap hashing).
    key = tick*1e12 + (cellX + 50000)*1e6 + (cellY + 50000);
    [gIdx_sorted, order] = sort(key);
    boundaries = [0; find(diff(gIdx_sorted)); numel(gIdx_sorted)];
    nG = numel(boundaries) - 1;

    nConflicts = 0;
    counted = containers.Map('KeyType','int64','ValueType','logical');

    for gg = 1:nG
        idxs = order(boundaries(gg)+1 : boundaries(gg+1));
        if numel(idxs) < 2, continue; end
        for a = 1:numel(idxs)
            for b = a+1:numel(idxs)
                idA = allID(idxs(a));
                idB = allID(idxs(b));
                if idA == idB, continue; end
                dH = hypot(allX(idxs(a))-allX(idxs(b)), ...
                           allY(idxs(a))-allY(idxs(b)));
                dV = abs(allZ(idxs(a)) - allZ(idxs(b)));
                if dH < cfg.conflict.hThresholdM && dV < vThr
                    pairKey = int64(min(idA,idB))*1e7 + ...
                              int64(max(idA,idB))*10 + ...
                              int64(mod(tick(idxs(a)), 1e5));
                    if ~isKey(counted, pairKey)
                        counted(pairKey) = true;
                        nConflicts = nConflicts + 1;
                    end
                end
            end
        end
    end
end


function fig = makeResultsFigure(agg, topoParams, network, depots, ...
    restaurants, residences, cfg)
% Six-panel publication figure.

    fig = figure('Position', [100 100 1400 900], 'Color','w');

    % ---- panel 1: spatial layout -------------------------------------
    subplot(2,3,1); hold on; grid on;
    plot(network.G, 'XData', network.nodeX, 'YData', network.nodeY, ...
        'EdgeColor', [0.85 0.85 0.85], 'NodeColor', 'none', 'LineWidth', 0.5);
    scatter(residences.x, residences.y, 8, [0.2 0.6 1.0], 'filled', ...
        'MarkerFaceAlpha', 0.35);
    scatter(restaurants.x, restaurants.y, 18, [1.0 0.4 0.2], 'filled');
    scatter(depots.x, depots.y, 180, 'k', 'p', 'filled');
    axis equal tight;
    xlabel('Easting (m)'); ylabel('Northing (m)');
    title('SF study area: depots (\bigstar), restaurants, residences');
    legend({'streets','residences','restaurants','depots'}, ...
        'Location','southoutside','Orientation','horizontal');

    % ---- panel 2: conflicts vs drones -------------------------------
    subplot(2,3,2); hold on; grid on;
    plotByTopology(agg, 'mean_conflictsPer1k', 'std_conflictsPer1k');
    yline(5,'--','acceptable threshold','Color',[0.4 0.4 0.4]);
    xlabel('Fleet size (drones)'); ylabel('Conflicts per 1000 flights');
    title('Capacity curve');  legend('Location','northwest');

    % ---- panel 3: flight time vs drones ------------------------------
    subplot(2,3,3); hold on; grid on;
    plotByTopology(agg, 'mean_avgFlightTimeS', 'std_avgFlightTimeS');
    xlabel('Fleet size (drones)'); ylabel('Avg flight time (s)');
    title('Flight time'); legend('Location','northwest');

    % ---- panel 4: detour ratio --------------------------------------
    subplot(2,3,4); hold on; grid on;
    plotByTopology(agg, 'mean_avgDetourRatio', 'std_avgDetourRatio');
    yline(sqrt(2),':','\surd 2 grid penalty','Color',[0.4 0.4 0.4]);
    xlabel('Fleet size (drones)'); ylabel('Path / Euclidean');
    title('Detour ratio');  legend('Location','northwest');

    % ---- panel 5: energy --------------------------------------------
    subplot(2,3,5); hold on; grid on;
    plotByTopology(agg, 'mean_avgEnergyWh', 'std_avgEnergyWh');
    xlabel('Fleet size (drones)'); ylabel('Avg energy per mission (Wh)');
    title('Energy usage'); legend('Location','northwest');

    % ---- panel 6: Pareto frontier -----------------------------------
    subplot(2,3,6); hold on; grid on;
    topos = unique(agg.topology, 'stable');
    colors = lines(numel(topos));
    for i = 1:numel(topos)
        sub = agg(agg.topology == topos(i),:);
        scatter(sub.mean_conflictsPer1k, sub.mean_avgEnergyWh, ...
            80, colors(i,:), 'filled', 'DisplayName', char(topos(i)));
        % annotate with fleet size
        for j = 1:height(sub)
            text(sub.mean_conflictsPer1k(j)+0.2, sub.mean_avgEnergyWh(j), ...
                sprintf('%d',sub.nDrones(j)), 'FontSize',8, 'Color', colors(i,:));
        end
    end
    xlabel('Conflicts per 1000 flights'); ylabel('Avg energy (Wh)');
    title('Safety-energy Pareto frontier'); legend('Location','best');

    sgtitle({'Decentralized Drone Delivery: Grid vs. Diagonal Overlay', ...
             sprintf('SF Mission/SoMa/Downtown - %d missions per run (avg)', ...
                 round(cfg.demand.simDurationS * cfg.demand.arrivalLambda(100) * 100))}, ...
        'FontWeight','bold','FontSize',12);
end


function plotByTopology(agg, meanCol, stdCol)
    topos = unique(agg.topology, 'stable');
    colors = lines(numel(topos));
    markers = {'o','s','^','d'};
    for i = 1:numel(topos)
        sub = agg(agg.topology == topos(i), :);
        sub = sortrows(sub, 'nDrones');
        errorbar(sub.nDrones, sub.(meanCol), sub.(stdCol), ...
            ['-' markers{mod(i-1,4)+1}], ...
            'Color', colors(i,:), 'MarkerFaceColor', colors(i,:), ...
            'LineWidth', 1.5, 'MarkerSize', 7, ...
            'DisplayName', char(topos(i)));
    end
end


function traj = emptyTraj()
    traj.t = []; traj.x = []; traj.y = []; traj.z = []; traj.band = {};
    traj.totalTimeS = 0; traj.pathM = 0; traj.detourRatio = 1;
    traj.energyWh = 0; traj.startT = 0; traj.endT = 0;
end

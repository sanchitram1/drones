%% DRONE GRID DELIVERY SIMULATION
%  Decentralized drone routing on a street grid with altitude-separated lanes
%  Course: Transportation Optimization
%  -----------------------------------------------------------------------
%  System Rules (fully decentralized — each drone follows these locally):
%    1. Fly along street-grid corridors only (no diagonal shortcuts)
%    2. Altitude is determined by compass heading (no central assignment)
%    3. Turns happen only at intersections, with a climb-turn-descend protocol
%    4. Right-of-way at intersections is resolved by altitude priority
%  -----------------------------------------------------------------------

clear; clc; close all;

%% ===== 1. SIMULATION PARAMETERS ========================================

% --- Grid definition ---
grid_size    = 10;            % 10x10 blocks (think 10 city blocks in each dir)
block_length = 200;           % meters per block edge
n_intersections = (grid_size+1)^2;  % total intersection nodes

% --- Altitude scheme (meters AGL) ---
% Key insight: separate EVERY direction to its own altitude band so that
% two drones on crossing streets never share the same height.
% We also separate opposing directions (N vs S) to eliminate head-on risk.
alt.north = 50;    % northbound traffic
alt.south = 60;    % southbound traffic  (10 m above northbound)
alt.east  = 70;    % eastbound traffic
alt.west  = 80;    % westbound traffic   (10 m above eastbound)
alt.transition = 95; % intersection transition ceiling (above all lanes)
alt.ground = 0;     % takeoff / landing

% Minimum vertical separation between lanes = 10 m, which exceeds the
% typical 3-5 m rotor wash / GPS-error envelope for small delivery drones.

% --- Drone parameters ---
cruise_speed   = 15;    % m/s horizontal cruise (~54 km/h, typical for delivery)
climb_rate     = 3;     % m/s vertical climb
descend_rate   = 2;     % m/s vertical descend (slower for stability & noise)
turn_time      = 4;     % seconds to execute a 90-deg turn at transition alt
sensor_range   = 50;    % meters — local detect-and-avoid radius

% --- Mission parameters ---
n_drones       = 50;    % number of simultaneous delivery missions
rng(42);                % reproducibility

%% ===== 2. BUILD THE GRID GRAPH =========================================

% Intersection coordinates (row, col) → (x, y) in meters
[col_idx, row_idx] = meshgrid(0:grid_size, 0:grid_size);
node_x = col_idx(:) * block_length;   % x coords of all intersections
node_y = row_idx(:) * block_length;   % y coords of all intersections
n_nodes = length(node_x);

% Node ID helper: node_id(row, col) — 1-indexed, row/col in 1..grid_size+1
node_id = @(r,c) (r-1)*(grid_size+1) + c;

% Build adjacency for the grid (undirected edges, Manhattan connectivity)
edges = [];
for r = 1:(grid_size+1)
    for c = 1:(grid_size+1)
        id = node_id(r,c);
        if c < grid_size+1   % east neighbor
            edges = [edges; id, node_id(r, c+1)];
        end
        if r < grid_size+1   % north neighbor
            edges = [edges; id, node_id(r+1, c)];
        end
    end
end

% Create graph object for shortest-path routing
G = graph(edges(:,1), edges(:,2));

fprintf('Grid: %dx%d blocks | %d intersections | %d edges\n', ...
    grid_size, grid_size, n_nodes, size(edges,1));

%% ===== 3. GENERATE RANDOM MISSIONS =====================================

origins      = randi(n_nodes, n_drones, 1);
destinations = randi(n_nodes, n_drones, 1);
% origins      = ones(n_nodes, n_drones) * 5;
% destinations = ones(n_nodes, n_drones) * 4;

% Make sure origin ≠ destination
same = (origins == destinations);
while any(same)
    destinations(same) = randi(n_nodes, sum(same), 1);
    same = (origins == destinations);
end

%% ===== 4. ROUTE EACH DRONE (SHORTEST MANHATTAN PATH) ===================

routes = cell(n_drones, 1);
manhattan_dist   = zeros(n_drones, 1);
euclidean_dist   = zeros(n_drones, 1);
total_flight_time = zeros(n_drones, 1);
n_turns          = zeros(n_drones, 1);

for d = 1:n_drones
    % Shortest path on the grid graph (BFS on unweighted = Manhattan optimal)
    path_nodes = shortestpath(G, origins(d), destinations(d));
    routes{d}  = path_nodes;

    % --- Compute distances ---
    path_x = node_x(path_nodes);
    path_y = node_y(path_nodes);

    seg_dx = diff(path_x);
    seg_dy = diff(path_y);
    seg_len = abs(seg_dx) + abs(seg_dy);          % each segment is axis-aligned
    manhattan_dist(d) = sum(seg_len);

    euclidean_dist(d) = sqrt((path_x(end)-path_x(1))^2 + ...
                             (path_y(end)-path_y(1))^2);

    % --- Assign altitudes per segment and count turns ---
    seg_alt  = zeros(length(path_nodes)-1, 1);
    seg_dir  = strings(length(path_nodes)-1, 1);   % direction label
    turns_this_drone = 0;

    for s = 1:length(path_nodes)-1
        dx = seg_dx(s); dy = seg_dy(s);
        if     dy > 0;  seg_alt(s) = alt.north; seg_dir(s) = "N";
        elseif dy < 0;  seg_alt(s) = alt.south; seg_dir(s) = "S";
        elseif dx > 0;  seg_alt(s) = alt.east;  seg_dir(s) = "E";
        elseif dx < 0;  seg_alt(s) = alt.west;  seg_dir(s) = "W";
        end

        if s > 1 && seg_dir(s) ~= seg_dir(s-1)
            turns_this_drone = turns_this_drone + 1;
        end
    end
    n_turns(d) = turns_this_drone;

    % --- Flight time estimate ---
    %  Takeoff:  ground → first segment altitude
    %  Cruise:   horizontal at assigned altitude
    %  Turn:     climb to transition alt → turn → descend to new alt
    %  Landing:  last segment altitude → ground

    t_takeoff = seg_alt(1) / climb_rate;
    t_land    = seg_alt(end) / descend_rate;
    t_cruise  = manhattan_dist(d) / cruise_speed;

    % Turn time: for each turn we climb from current alt to transition,
    % execute the turn, then descend to new alt
    t_turns = 0;
    for s = 2:length(seg_alt)
        if seg_dir(s) ~= seg_dir(s-1)
            climb_up   = (alt.transition - seg_alt(s-1)) / climb_rate;
            turn_exec  = turn_time;
            descend_dn = (alt.transition - seg_alt(s)) / descend_rate;
            t_turns = t_turns + climb_up + turn_exec + descend_dn;
        end
    end

    total_flight_time(d) = t_takeoff + t_cruise + t_turns + t_land;
end

%% ===== 5. CONFLICT / CAPACITY ANALYSIS =================================

% We discretize time into 1-second steps and check how many drones occupy
% the same grid edge in the same second. A "conflict" = two drones on the
% same edge + same direction band within sensor_range at the same time.

sim_duration = ceil(max(total_flight_time)) + 60;  % seconds
dt = 1;  % time step

% For each drone, build a timetable: (time, edge_id, altitude)
% Simplified: we assume drones depart at staggered times (uniform random
% within a launch window) to approximate real-world arrivals.
launch_window = 120;  % seconds — all drones launch within this window
launch_times  = rand(n_drones,1) * launch_window;

% Track edge occupancy: key = "edge_nodeA_nodeB", value = list of (drone, time)
edge_occupancy = containers.Map();

for d = 1:n_drones
    path_nodes = routes{d};
    seg_dx = diff(node_x(path_nodes));
    seg_dy = diff(node_y(path_nodes));

    cum_time = launch_times(d);

    % Takeoff
    first_alt = 0;
    dx1 = seg_dx(1); dy1 = seg_dy(1);
    if     dy1 > 0; first_alt = alt.north;
    elseif dy1 < 0; first_alt = alt.south;
    elseif dx1 > 0; first_alt = alt.east;
    elseif dx1 < 0; first_alt = alt.west;
    end
    cum_time = cum_time + first_alt / climb_rate;

    prev_dir = "";
    for s = 1:length(path_nodes)-1
        dx = seg_dx(s); dy = seg_dy(s);
        if     dy > 0; cur_dir = "N"; cur_alt = alt.north;
        elseif dy < 0; cur_dir = "S"; cur_alt = alt.south;
        elseif dx > 0; cur_dir = "E"; cur_alt = alt.east;
        elseif dx < 0; cur_dir = "W"; cur_alt = alt.west;
        end

        % Turn penalty
        if prev_dir ~= "" && cur_dir ~= prev_dir
            cum_time = cum_time + ...
                (alt.transition - cur_alt)/climb_rate + ...
                turn_time + ...
                (alt.transition - cur_alt)/descend_rate;
        end

        seg_travel = block_length / cruise_speed;
        enter_time = cum_time;
        exit_time  = cum_time + seg_travel;

        % Record occupancy
        n1 = min(path_nodes(s), path_nodes(s+1));
        n2 = max(path_nodes(s), path_nodes(s+1));
        edge_key = sprintf('%d_%d', n1, n2);

        if ~edge_occupancy.isKey(edge_key)
            edge_occupancy(edge_key) = [];
        end
        entry = struct('drone', d, 'enter', enter_time, ...
                       'exit', exit_time, 'alt', cur_alt, 'dir', cur_dir);
        edge_occupancy(edge_key) = [edge_occupancy(edge_key), entry];

        cum_time = exit_time;
        prev_dir = cur_dir;
    end
end

% Count pairwise conflicts on each edge
n_conflicts = 0;
conflict_details = [];

all_keys = keys(edge_occupancy);
for k = 1:length(all_keys)
    entries = edge_occupancy(all_keys{k});
    for i = 1:length(entries)
        for j = i+1:length(entries)
            % Same altitude band AND overlapping time window
            if entries(i).alt == entries(j).alt
                overlap = min(entries(i).exit, entries(j).exit) - ...
                          max(entries(i).enter, entries(j).enter);
                if overlap > 0
                    % Same direction = trailing conflict (less severe)
                    % Opposite direction at same alt shouldn't happen with
                    % our 4-altitude scheme, but flag it anyway
                    n_conflicts = n_conflicts + 1;
                    conflict_details = [conflict_details; ...
                        struct('edge', all_keys{k}, ...
                               'drone_a', entries(i).drone, ...
                               'drone_b', entries(j).drone, ...
                               'overlap_sec', overlap, ...
                               'altitude', entries(i).alt)];
                end
            end
        end
    end
end

% Edge utilization (max simultaneous drones per edge at any altitude)
max_edge_load = 0;
busiest_edge  = '';
for k = 1:length(all_keys)
    entries = edge_occupancy(all_keys{k});
    n_entries = length(entries);
    if n_entries > max_edge_load
        max_edge_load = n_entries;
        busiest_edge = all_keys{k};
    end
end

%% ===== 6. PRINT SUMMARY STATISTICS =====================================

fprintf('\n========== SIMULATION RESULTS ==========\n');
fprintf('Drones simulated    : %d\n', n_drones);
fprintf('Launch window       : %.0f seconds\n', launch_window);
fprintf('Grid                : %d x %d blocks (%d m each)\n', ...
    grid_size, grid_size, block_length);
fprintf('\n--- Distance Metrics ---\n');
fprintf('Avg Manhattan dist  : %.0f m\n', mean(manhattan_dist));
fprintf('Avg Euclidean dist  : %.0f m\n', mean(euclidean_dist));
fprintf('Avg detour ratio    : %.2fx  (Manhattan / Euclidean)\n', ...
    mean(manhattan_dist ./ euclidean_dist));
fprintf('Max detour ratio    : %.2fx\n', max(manhattan_dist ./ euclidean_dist));
fprintf('\n--- Flight Time ---\n');
fprintf('Avg flight time     : %.1f s  (%.1f min)\n', ...
    mean(total_flight_time), mean(total_flight_time)/60);
fprintf('Max flight time     : %.1f s  (%.1f min)\n', ...
    max(total_flight_time), max(total_flight_time)/60);
fprintf('Avg turns per route : %.1f\n', mean(n_turns));
fprintf('\n--- Capacity & Conflicts ---\n');
fprintf('Total conflicts     : %d  (same edge + same alt + overlapping time)\n', n_conflicts);
fprintf('Busiest edge load   : %d drones (edge %s)\n', max_edge_load, busiest_edge);
fprintf('Altitude bands used : N=%dm  S=%dm  E=%dm  W=%dm  Trans=%dm\n', ...
    alt.north, alt.south, alt.east, alt.west, alt.transition);

%% ===== 7. VISUALIZATIONS ===============================================

figure('Position', [50 50 1600 900], 'Name', 'Drone Grid Simulation');

% --- 7a. Top-down route map ---
subplot(2,3,1);
hold on; axis equal; grid on;
title('Top-Down Route Map', 'FontSize', 12);
xlabel('x (m)'); ylabel('y (m)');

% Draw grid streets
for r = 0:grid_size
    plot([0, grid_size*block_length], [r*block_length, r*block_length], ...
        '-', 'Color', [0.85 0.85 0.85], 'LineWidth', 0.5);
end
for c = 0:grid_size
    plot([c*block_length, c*block_length], [0, grid_size*block_length], ...
        '-', 'Color', [0.85 0.85 0.85], 'LineWidth', 0.5);
end

% Color routes by drone index
cmap = parula(n_drones);
for d = 1:n_drones
    px = node_x(routes{d});
    py = node_y(routes{d});
    plot(px, py, '-', 'Color', [cmap(d,:), 0.4], 'LineWidth', 1.2);
    plot(px(1), py(1), 'go', 'MarkerSize', 4, 'MarkerFaceColor', 'g');
    plot(px(end), py(end), 'rs', 'MarkerSize', 4, 'MarkerFaceColor', 'r');
end
legend({'', '', 'Origin', 'Destination'}, 'Location', 'best');

% --- 7b. 3D trajectory for first 5 drones ---
subplot(2,3,2);
hold on; grid on; view(30, 25);
title('3D Trajectories (First 5 Drones)', 'FontSize', 12);
xlabel('x (m)'); ylabel('y (m)'); zlabel('Altitude (m)');

colors_3d = lines(5);
for d = 1:min(5, n_drones)
    path_nodes = routes{d};
    px = node_x(path_nodes);
    py = node_y(path_nodes);
    seg_dx_3d = diff(px);
    seg_dy_3d = diff(py);

    % Build 3D waypoints: takeoff, segments with altitude, landing
    wx = [px(1)]; wy = [py(1)]; wz = [0];  % start on ground

    prev_d3 = "";
    for s = 1:length(path_nodes)-1
        dx = seg_dx_3d(s); dy = seg_dy_3d(s);
        if     dy > 0; cur_d3 = "N"; cur_a = alt.north;
        elseif dy < 0; cur_d3 = "S"; cur_a = alt.south;
        elseif dx > 0; cur_d3 = "E"; cur_a = alt.east;
        elseif dx < 0; cur_d3 = "W"; cur_a = alt.west;
        end

        if s == 1
            % Takeoff: climb to first segment altitude at origin
            wx = [wx, px(1)]; wy = [wy, py(1)]; wz = [wz, cur_a];
        elseif cur_d3 ~= prev_d3
            % Turn: climb to transition, then descend to new alt
            wx = [wx, px(s)]; wy = [wy, py(s)]; wz = [wz, alt.transition];
            wx = [wx, px(s)]; wy = [wy, py(s)]; wz = [wz, cur_a];
        end

        % End of segment
        wx = [wx, px(s+1)]; wy = [wy, py(s+1)]; wz = [wz, cur_a];
        prev_d3 = cur_d3;
    end

    % Landing
    wx = [wx, px(end)]; wy = [wy, py(end)]; wz = [wz, 0];

    plot3(wx, wy, wz, '-', 'Color', colors_3d(d,:), 'LineWidth', 1.5);
    plot3(wx(1), wy(1), wz(1), 'go', 'MarkerSize', 6, 'MarkerFaceColor', 'g');
    plot3(wx(end), wy(end), wz(end), 'rs', 'MarkerSize', 6, 'MarkerFaceColor', 'r');
end

% Draw altitude reference planes
[Xp, Yp] = meshgrid(linspace(0, grid_size*block_length, 3));
alpha_val = 0.05;
for a = [alt.north, alt.south, alt.east, alt.west]
    surf(Xp, Yp, a*ones(size(Xp)), 'FaceAlpha', alpha_val, ...
        'EdgeColor', 'none', 'FaceColor', [0.5 0.5 1]);
end

% --- 7c. Manhattan vs Euclidean scatter ---
subplot(2,3,3);
scatter(euclidean_dist, manhattan_dist, 30, n_turns, 'filled', 'MarkerFaceAlpha', 0.7);
hold on;
max_d = max([euclidean_dist; manhattan_dist]) * 1.1;
plot([0 max_d], [0 max_d], 'k--', 'LineWidth', 0.8);         % y = x
plot([0 max_d], [0 max_d*sqrt(2)], 'r--', 'LineWidth', 0.8); % worst case grid
colorbar; colormap(gca, 'hot');
xlabel('Euclidean Distance (m)'); ylabel('Manhattan Distance (m)');
title('Manhattan vs Euclidean', 'FontSize', 12);
legend('Drones', 'y=x (direct)', 'y=\surd2 x (worst grid)', 'Location', 'best');
caxis([0 max(n_turns)]);
ylabel(colorbar, 'Number of Turns');

% --- 7d. Altitude band utilization ---
subplot(2,3,4);
dir_labels = {'North','South','East','West'};
alt_vals   = [alt.north, alt.south, alt.east, alt.west];
seg_counts = zeros(4,1);  % count total segments per direction
for d = 1:n_drones
    path_nodes = routes{d};
    seg_dx_c = diff(node_x(path_nodes));
    seg_dy_c = diff(node_y(path_nodes));
    for s = 1:length(seg_dx_c)
        if     seg_dy_c(s) > 0; seg_counts(1) = seg_counts(1) + 1;
        elseif seg_dy_c(s) < 0; seg_counts(2) = seg_counts(2) + 1;
        elseif seg_dx_c(s) > 0; seg_counts(3) = seg_counts(3) + 1;
        elseif seg_dx_c(s) < 0; seg_counts(4) = seg_counts(4) + 1;
        end
    end
end
bar(alt_vals, seg_counts, 0.5, 'FaceColor', [0.3 0.6 0.9]);
xlabel('Altitude (m)'); ylabel('Total Segments Used');
title('Altitude Band Utilization', 'FontSize', 12);
set(gca, 'XTick', alt_vals, 'XTickLabel', ...
    arrayfun(@(a,i) sprintf('%s\n%dm', dir_labels{i}, a), alt_vals, 1:4, 'Uni', 0));

% --- 7e. Flight time distribution ---
subplot(2,3,5);
histogram(total_flight_time/60, 15, 'FaceColor', [0.9 0.4 0.3], 'FaceAlpha', 0.8);
xlabel('Flight Time (min)'); ylabel('Count');
title('Flight Time Distribution', 'FontSize', 12);
xline(mean(total_flight_time)/60, 'b--', sprintf('Mean: %.1f min', mean(total_flight_time)/60), ...
    'LineWidth', 1.5, 'LabelOrientation', 'horizontal');

% --- 7f. Detour ratio histogram ---
subplot(2,3,6);
detour = manhattan_dist ./ euclidean_dist;
histogram(detour, 15, 'FaceColor', [0.4 0.8 0.5], 'FaceAlpha', 0.8);
xlabel('Detour Ratio (Manhattan / Euclidean)'); ylabel('Count');
title('Grid Penalty Distribution', 'FontSize', 12);
xline(mean(detour), 'b--', sprintf('Mean: %.2f', mean(detour)), ...
    'LineWidth', 1.5, 'LabelOrientation', 'horizontal');
xline(sqrt(2), 'r--', sprintf('Theoretical max: %.2f', sqrt(2)), ...
    'LineWidth', 1.2, 'LabelOrientation', 'horizontal');

sgtitle('Decentralized Drone Grid Delivery — Baseline Simulation', 'FontSize', 14, 'FontWeight', 'bold');

%% ===== 8. CAPACITY SWEEP: HOW MANY DRONES CAN THE GRID HANDLE? ========

fprintf('\n\n========== CAPACITY SWEEP ==========\n');
drone_counts = [25, 50, 100, 200, 400, 800];
conflict_results = zeros(length(drone_counts), 1);
avg_times        = zeros(length(drone_counts), 1);

for trial = 1:length(drone_counts)
    nd = drone_counts(trial);
    o = randi(n_nodes, nd, 1);
    dest = randi(n_nodes, nd, 1);
    same_od = (o == dest);
    while any(same_od)
        dest(same_od) = randi(n_nodes, sum(same_od), 1);
        same_od = (o == dest);
    end

    lt = rand(nd,1) * launch_window;
    eo = containers.Map();
    ft = zeros(nd,1);

    for d = 1:nd
        pn = shortestpath(G, o(d), dest(d));
        px_ = node_x(pn); py_ = node_y(pn);
        sdx = diff(px_); sdy = diff(py_);

        ct = lt(d);
        % Takeoff
        dy1 = sdy(1); dx1 = sdx(1);
        if     dy1 > 0; fa = alt.north;
        elseif dy1 < 0; fa = alt.south;
        elseif dx1 > 0; fa = alt.east;
        else;           fa = alt.west;
        end
        ct = ct + fa/climb_rate;

        pd = "";
        for s = 1:length(pn)-1
            dx = sdx(s); dy = sdy(s);
            if     dy > 0; cd = "N"; ca = alt.north;
            elseif dy < 0; cd = "S"; ca = alt.south;
            elseif dx > 0; cd = "E"; ca = alt.east;
            else;          cd = "W"; ca = alt.west;
            end
            if pd ~= "" && cd ~= pd
                ct = ct + (alt.transition-ca)/climb_rate + turn_time + (alt.transition-ca)/descend_rate;
            end
            st = block_length/cruise_speed;
            et = ct; xt = ct + st;
            n1 = min(pn(s), pn(s+1)); n2 = max(pn(s), pn(s+1));
            ek = sprintf('%d_%d', n1, n2);
            if ~eo.isKey(ek); eo(ek) = []; end
            entry = struct('drone',d,'enter',et,'exit',xt,'alt',ca,'dir',cd);
            eo(ek) = [eo(ek), entry];
            ct = xt; pd = cd;
        end
        ct = ct + ca/descend_rate;
        ft(d) = ct - lt(d);
    end

    nc = 0;
    ak = keys(eo);
    for k = 1:length(ak)
        en = eo(ak{k});
        for i = 1:length(en)
            for j = i+1:length(en)
                if en(i).alt == en(j).alt
                    ol = min(en(i).exit, en(j).exit) - max(en(i).enter, en(j).enter);
                    if ol > 0; nc = nc+1; end
                end
            end
        end
    end

    conflict_results(trial) = nc;
    avg_times(trial) = mean(ft);
    fprintf('  %4d drones → %4d conflicts | avg time %.1f s\n', ...
        nd, nc, avg_times(trial));
end

figure('Position', [100 100 800 400], 'Name', 'Capacity Analysis');
subplot(1,2,1);
plot(drone_counts, conflict_results, 'ro-', 'LineWidth', 2, 'MarkerFaceColor', 'r');
xlabel('Number of Drones'); ylabel('Conflicts');
title('Conflicts vs Drone Count'); grid on;
subplot(1,2,2);
plot(drone_counts, avg_times/60, 'bo-', 'LineWidth', 2, 'MarkerFaceColor', 'b');
xlabel('Number of Drones'); ylabel('Avg Flight Time (min)');
title('Avg Flight Time vs Drone Count'); grid on;
sgtitle('Grid Capacity Analysis', 'FontSize', 13, 'FontWeight', 'bold');

%% ===== 9. DIAGONAL SHORTCUT ANALYSIS (PREVIEW) =========================
%  Compare Manhattan grid vs. hypothetical diagonal routes.
%  Here we just compute the theoretical savings — the full diagonal
%  simulation would be a separate module.

fprintf('\n\n========== DIAGONAL vs GRID COMPARISON ==========\n');
savings_pct = (1 - euclidean_dist ./ manhattan_dist) * 100;
fprintf('If drones could fly direct (Euclidean):\n');
fprintf('  Avg distance saving : %.1f%%\n', mean(savings_pct));
fprintf('  Max distance saving : %.1f%%  (diagonal OD pair)\n', max(savings_pct));
fprintf('  Theoretical max saving on grid: %.1f%%  (1 - 1/sqrt(2))\n', ...
    (1 - 1/sqrt(2))*100);
fprintf('\nNote: Diagonal routes eliminate turns but require:\n');
fprintf('  - Flying over buildings (noise, privacy, safety)\n');
fprintf('  - More complex altitude deconfliction (infinite headings)\n');
fprintf('  - Loss of building-canyon wind protection\n');

fprintf('\n========== SIMULATION COMPLETE ==========\n');
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd


SIM_TURN_PROTOCOL_CHOICES = (
    "simple",
    "turn_layer",
    "intersection_cube",
    "sphereabout",
)
OPTIMIZER_PROTOCOL_CHOICES = (
    "turn_layer",
    "intersection_cube",
    "sphereabout",
)
TOPOLOGY_CHOICES = ("grid", "diagonal_overlay")
SIM_DEMAND_MODEL_CHOICES = ("time-series", "fixed-count", "csv")
OPTIMIZE_DEMAND_MODEL_CHOICES = ("simulator", "gravity", "uniform")


def normalize_choice(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def parse_turn_protocol(value: str) -> str:
    normalized = normalize_choice(value)
    if normalized not in SIM_TURN_PROTOCOL_CHOICES:
        choices = ", ".join(SIM_TURN_PROTOCOL_CHOICES)
        raise argparse.ArgumentTypeError(
            f"Unsupported turn protocol '{value}'. Choose from: {choices}."
        )
    return normalized


def parse_optimizer_protocol(value: str) -> str:
    normalized = normalize_choice(value)
    if normalized not in OPTIMIZER_PROTOCOL_CHOICES:
        choices = ", ".join(OPTIMIZER_PROTOCOL_CHOICES)
        raise argparse.ArgumentTypeError(
            f"Unsupported optimizer protocol '{value}'. Choose from: {choices}."
        )
    return normalized


def parse_demand_model(value: str) -> str:
    normalized = normalize_choice(value).replace("_", "-")
    if normalized not in SIM_DEMAND_MODEL_CHOICES:
        choices = ", ".join(SIM_DEMAND_MODEL_CHOICES)
        raise argparse.ArgumentTypeError(
            f"Unsupported demand model '{value}'. Choose from: {choices}."
        )
    return normalized


def parse_optimizer_demand_model(value: str) -> str:
    normalized = normalize_choice(value)
    if normalized not in OPTIMIZE_DEMAND_MODEL_CHOICES:
        choices = ", ".join(OPTIMIZE_DEMAND_MODEL_CHOICES)
        raise argparse.ArgumentTypeError(
            f"Unsupported optimizer demand model '{value}'. Choose from: {choices}."
        )
    return normalized


def parse_optional_column(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    if text.lower() in {"", "none", "null"}:
        return None
    return text


def parse_config_spec(value: str) -> tuple[str, str]:
    try:
        topology, protocol = value.split(":", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Configuration must look like 'grid:turn_layer'."
        ) from exc

    topology = normalize_choice(topology)
    protocol = parse_optimizer_protocol(protocol)
    if topology not in TOPOLOGY_CHOICES:
        choices = ", ".join(TOPOLOGY_CHOICES)
        raise argparse.ArgumentTypeError(
            f"Unsupported topology '{topology}'. Choose from: {choices}."
        )
    return topology, protocol


def parse_k_max_values(value: str) -> list[float]:
    pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
    if not pieces:
        raise argparse.ArgumentTypeError(
            "K-max values must be a comma-separated list like '10,25,50,100'."
        )
    try:
        values = [float(piece) for piece in pieces]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "K-max values must be numeric."
        ) from exc
    return values


def resolve_output_prefix(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.suffix:
        path = path.with_suffix("")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_json(payload: dict, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return str(path)


def save_dataframe(df: pd.DataFrame, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return str(path)


def save_figure(fig, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return str(path)


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing sf_nodes.csv, sf_edges.csv, sf_census.csv, and sf_restaurants.csv.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for demand generation and simulation.",
    )
    parser.add_argument(
        "--turn-protocol",
        type=parse_turn_protocol,
        default="simple",
        help=(
            "Turning protocol to use. 'simple' and 'turn_layer' are currently wired "
            "to the same behavior in this engine; the others are accepted but will "
            "raise until that implementation lands."
        ),
    )
    parser.add_argument(
        "--orders-csv",
        type=Path,
        help="Path to an orders CSV with request_time_s, origin_node, and dest_node columns.",
    )
    parser.add_argument(
        "--demand-model",
        type=parse_demand_model,
        default="time-series",
        help="Demand generation mode: time-series, fixed-count, or csv.",
    )
    parser.add_argument(
        "--n-orders",
        type=int,
        help="Number of orders for fixed-count demand.",
    )
    parser.add_argument(
        "--sim-duration-min",
        type=float,
        default=120.0,
        help="Simulation horizon in minutes.",
    )
    parser.add_argument(
        "--dt-s",
        type=int,
        default=30,
        help="Demand discretization step in seconds.",
    )
    parser.add_argument(
        "--base-lambda-per-min",
        type=float,
        default=6.0,
        help="Base arrival rate in orders per minute for time-series demand.",
    )
    parser.add_argument(
        "--peak-multiplier",
        type=float,
        default=1.8,
        help="Multiplier applied during the peak demand window.",
    )
    parser.add_argument(
        "--peak-start-min",
        type=float,
        default=45.0,
        help="Peak-window start time in minutes from simulation start.",
    )
    parser.add_argument(
        "--peak-end-min",
        type=float,
        default=95.0,
        help="Peak-window end time in minutes from simulation start.",
    )
    parser.add_argument(
        "--origin-weight-col",
        type=parse_optional_column,
        default=None,
        help="Optional restaurant column to weight origin sampling. Use 'none' for uniform sampling.",
    )
    parser.add_argument(
        "--dest-weight-col",
        type=parse_optional_column,
        default="pop_density",
        help="Census column to weight destination sampling. Use 'none' for uniform sampling.",
    )
    parser.add_argument(
        "--dest-jitter-m",
        type=float,
        default=100.0,
        help="Gaussian destination jitter in meters.",
    )
    parser.add_argument(
        "--cruise-speed-ft-s",
        type=float,
        default=35.0,
        help="Cruise speed in feet per second.",
    )
    parser.add_argument(
        "--climb-rate-ft-s",
        type=float,
        default=8.0,
        help="Climb rate in feet per second.",
    )
    parser.add_argument(
        "--descend-rate-ft-s",
        type=float,
        default=7.0,
        help="Descent rate in feet per second.",
    )
    parser.add_argument(
        "--turn-time-s",
        type=float,
        default=4.0,
        help="Time spent executing the turn maneuver, excluding climb and descent.",
    )
    parser.add_argument(
        "--edge-headway-s",
        type=float,
        default=3.0,
        help="Minimum headway on the same corridor and direction.",
    )
    parser.add_argument(
        "--intersection-headway-s",
        type=float,
        default=4.0,
        help="Minimum headway when reserving an intersection turn window.",
    )
    parser.add_argument(
        "--max-altitude-ft",
        type=float,
        default=400.0,
        help="Maximum allowed altitude in feet.",
    )
    parser.add_argument(
        "--building-clearance-ft",
        type=float,
        default=5.0,
        help="Additional building clearance buffer in feet.",
    )
    parser.add_argument(
        "--collision-radius-ft",
        type=float,
        default=10.0,
        help="Near-conflict screening radius in feet.",
    )
    parser.add_argument(
        "--alt-north-ft",
        type=float,
        default=300.0,
        help="Cruise altitude for northbound segments in feet.",
    )
    parser.add_argument(
        "--alt-south-ft",
        type=float,
        default=315.0,
        help="Cruise altitude for southbound segments in feet.",
    )
    parser.add_argument(
        "--alt-east-ft",
        type=float,
        default=330.0,
        help="Cruise altitude for eastbound segments in feet.",
    )
    parser.add_argument(
        "--alt-west-ft",
        type=float,
        default=345.0,
        help="Cruise altitude for westbound segments in feet.",
    )
    parser.add_argument(
        "--alt-turn-ft",
        type=float,
        default=370.0,
        help="Turn-layer altitude in feet.",
    )
    parser.add_argument(
        "--max-demo-routes-3d",
        type=int,
        default=8,
        help="Maximum number of routes shown in the 3D plot.",
    )
    parser.add_argument(
        "--max-demo-routes-map",
        type=int,
        default=120,
        help="Maximum number of routes drawn on the top-down map.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively.",
    )
    parser.add_argument(
        "--save-figure",
        type=Path,
        help="Base path or prefix for saved figures. Writes *_overview.png and *_summary.png.",
    )
    parser.add_argument(
        "--save-results",
        type=Path,
        help="Base path or prefix for saved CSV/JSON outputs.",
    )
    parser.add_argument(
        "--save-orders",
        type=Path,
        help="Write the resolved orders table to this CSV path.",
    )


def add_optimizer_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing the exported SF CSVs used by the simulator.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for OD sampling in the optimizer.",
    )
    parser.add_argument(
        "--demand-model",
        type=parse_optimizer_demand_model,
        default="simulator",
        help="Demand model for optimization: simulator, gravity, or uniform.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2000,
        help="Number of OD pairs sampled for the path distribution.",
    )
    parser.add_argument(
        "--K-max",
        type=float,
        default=50.0,
        help="Maximum tolerated conflicts per launch window.",
    )
    parser.add_argument(
        "--launch-window-s",
        type=float,
        default=300.0,
        help="Launch window used in the conflict-budget calculation.",
    )
    parser.add_argument(
        "--origin-weight-col",
        type=parse_optional_column,
        default=None,
        help="Restaurant column used to weight origin sampling in simulator demand mode.",
    )
    parser.add_argument(
        "--dest-weight-col",
        type=parse_optional_column,
        default="pop_density",
        help="Census column used to weight destination sampling in simulator demand mode.",
    )
    parser.add_argument(
        "--dest-jitter-m",
        type=float,
        default=100.0,
        help="Gaussian destination jitter in meters for simulator demand mode.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1e-4,
        help="Distance-decay coefficient for gravity demand mode.",
    )
    parser.add_argument(
        "--abstract",
        action="store_true",
        help="Use the abstract grid topology instead of the exported SF graph.",
    )
    parser.add_argument(
        "--enable-zone-capacity",
        action="store_true",
        help="Enable the optimizer's zone-capacity constraint.",
    )
    parser.add_argument(
        "--zone-capacity",
        type=int,
        default=50,
        help="Maximum drones allowed per zone when zone capacity is enabled.",
    )
    parser.add_argument(
        "--zone-grid-size",
        type=int,
        default=5,
        help="Zones per side for the zone-capacity model.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display optimizer figures interactively.",
    )
    parser.add_argument(
        "--save-figure",
        type=Path,
        help="Base path or prefix for saved optimizer figures.",
    )
    parser.add_argument(
        "--save-results",
        type=Path,
        help="Base path or prefix for saved optimizer CSV/JSON outputs.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="simulate",
        description="Run the drone simulation and analytical optimizer CLI.",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser(
        "run",
        help="Run the real-data simulation from drone_delivery_sf_realdata_visualization.py",
    )
    add_run_arguments(run_parser)
    run_parser.set_defaults(handler=handle_run)

    optimize_parser = subparsers.add_parser(
        "optimize",
        help="Run the analytical throughput optimizer.",
    )
    optimize_subparsers = optimize_parser.add_subparsers(dest="optimize_command")

    optimize_run_parser = optimize_subparsers.add_parser(
        "run",
        help="Run the optimizer for one configuration.",
    )
    optimize_run_parser.add_argument(
        "--config",
        type=parse_config_spec,
        required=True,
        help="Configuration spec like 'grid:turn_layer'.",
    )
    add_optimizer_common_arguments(optimize_run_parser)
    optimize_run_parser.set_defaults(handler=handle_optimize_run)

    optimize_compare_parser = optimize_subparsers.add_parser(
        "compare",
        help="Compare two or more optimizer configurations.",
    )
    optimize_compare_parser.add_argument(
        "--config",
        type=parse_config_spec,
        action="append",
        required=True,
        help="Configuration spec like 'grid:turn_layer'. Repeat this flag to compare multiple configurations.",
    )
    add_optimizer_common_arguments(optimize_compare_parser)
    optimize_compare_parser.set_defaults(handler=handle_optimize_compare)

    optimize_sensitivity_parser = optimize_subparsers.add_parser(
        "sensitivity",
        help="Sweep K_max and inspect which optimizer constraint binds.",
    )
    optimize_sensitivity_parser.add_argument(
        "--config",
        type=parse_config_spec,
        required=True,
        help="Configuration spec like 'grid:turn_layer'.",
    )
    add_optimizer_common_arguments(optimize_sensitivity_parser)
    optimize_sensitivity_parser.add_argument(
        "--K-max-values",
        type=parse_k_max_values,
        default=[10.0, 25.0, 50.0, 100.0, 200.0],
        help="Comma-separated list of conflict budgets to sweep.",
    )
    optimize_sensitivity_parser.set_defaults(handler=handle_optimize_sensitivity)

    return parser


def handle_run(args: argparse.Namespace) -> int:
    if not args.show:
        os.environ.setdefault("MPLBACKEND", "Agg")

    from drone_delivery_sf_realdata_visualization import SimConfig, run_simulation

    cfg = SimConfig(
        data_dir=str(args.data_dir),
        random_seed=args.seed,
        turn_protocol=args.turn_protocol,
        demand_model=args.demand_model,
        sim_duration_s=int(args.sim_duration_min * 60),
        dt_s=args.dt_s,
        base_lambda_per_min=args.base_lambda_per_min,
        peak_multiplier=args.peak_multiplier,
        peak_window_s=(
            int(args.peak_start_min * 60),
            int(args.peak_end_min * 60),
        ),
        n_orders_override=args.n_orders,
        origin_weight_col=args.origin_weight_col,
        dest_weight_col=args.dest_weight_col,
        dest_jitter_m=args.dest_jitter_m,
        cruise_speed_ft_s=args.cruise_speed_ft_s,
        climb_rate_ft_s=args.climb_rate_ft_s,
        descend_rate_ft_s=args.descend_rate_ft_s,
        turn_time_s=args.turn_time_s,
        max_altitude_ft=args.max_altitude_ft,
        building_clearance_ft=args.building_clearance_ft,
        collision_radius_ft=args.collision_radius_ft,
        alt_north_ft=args.alt_north_ft,
        alt_south_ft=args.alt_south_ft,
        alt_east_ft=args.alt_east_ft,
        alt_west_ft=args.alt_west_ft,
        alt_turn_ft=args.alt_turn_ft,
        edge_time_headway_s=args.edge_headway_s,
        intersection_headway_s=args.intersection_headway_s,
        max_demo_routes_3d=args.max_demo_routes_3d,
        max_demo_routes_map=args.max_demo_routes_map,
    )

    run_simulation(
        cfg=cfg,
        orders_csv=args.orders_csv,
        show=args.show,
        save_figure=args.save_figure,
        save_results=args.save_results,
        save_orders=args.save_orders,
    )
    return 0


def build_optimizer_config(config_spec: tuple[str, str], args: argparse.Namespace):
    from config import ExperimentConfig, SimConfig as OptimizerSimConfig, SFConfig

    topology, protocol = config_spec
    sim_cfg = OptimizerSimConfig(
        seed=args.seed,
        launch_window=float(args.launch_window_s),
        enable_admission_control=bool(args.enable_zone_capacity),
        zone_capacity=int(args.zone_capacity),
    )
    sf_cfg = SFConfig(zone_grid_size=int(args.zone_grid_size))
    return ExperimentConfig(
        topology=topology,
        turning_protocol=protocol,
        sim=sim_cfg,
        sf=sf_cfg,
        use_sf_data=not args.abstract,
    )


def handle_optimize_run(args: argparse.Namespace) -> int:
    if not args.show:
        os.environ.setdefault("MPLBACKEND", "Agg")

    from optimize import ThroughputOptimizer
    from optimize_viz import plot_optimization_result

    cfg = build_optimizer_config(args.config, args)
    optimizer = ThroughputOptimizer(cfg, data_dir=str(args.data_dir))
    result = optimizer.optimize(
        n_od_samples=args.samples,
        K_max=args.K_max,
        demand_mode=args.demand_model,
        origin_weight_col=args.origin_weight_col,
        dest_weight_col=args.dest_weight_col,
        dest_jitter_m=args.dest_jitter_m,
        beta=args.beta,
        verbose=True,
    )

    saved_paths = {}
    if args.save_results is not None:
        prefix = resolve_output_prefix(args.save_results)
        result_payload = asdict(result)
        result_payload["throughput_hr"] = result.throughput_per_hour()
        saved_paths["summary_json"] = save_json(
            result_payload,
            prefix.parent / f"{prefix.name}_summary.json",
        )
        saved_paths["summary_csv"] = save_dataframe(
            pd.DataFrame([result_payload]),
            prefix.parent / f"{prefix.name}_summary.csv",
        )

    if args.show or args.save_figure is not None:
        fig = plot_optimization_result(result)
        if args.save_figure is not None:
            prefix = resolve_output_prefix(args.save_figure)
            saved_paths["figure_png"] = save_figure(
                fig,
                prefix.parent / f"{prefix.name}_run.png",
            )
        if args.show:
            import matplotlib.pyplot as plt

            plt.show()
        else:
            import matplotlib.pyplot as plt

            plt.close(fig)

    for label, path in saved_paths.items():
        print(f"Saved {label}: {path}")
    return 0


def handle_optimize_compare(args: argparse.Namespace) -> int:
    if len(args.config) < 2:
        raise SystemExit("simulate optimize compare requires at least two --config values.")
    if not args.show:
        os.environ.setdefault("MPLBACKEND", "Agg")

    from optimize import compare_configuration_list
    from optimize_viz import plot_optimization_comparison

    configs = [build_optimizer_config(spec, args) for spec in args.config]
    df = compare_configuration_list(
        configs=configs,
        n_od_samples=args.samples,
        K_max=args.K_max,
        demand_mode=args.demand_model,
        origin_weight_col=args.origin_weight_col,
        dest_weight_col=args.dest_weight_col,
        dest_jitter_m=args.dest_jitter_m,
        beta=args.beta,
        data_dir=str(args.data_dir),
        verbose=True,
    )
    print("\n" + df.to_string(index=False))

    saved_paths = {}
    if args.save_results is not None:
        prefix = resolve_output_prefix(args.save_results)
        saved_paths["comparison_csv"] = save_dataframe(
            df,
            prefix.parent / f"{prefix.name}_compare.csv",
        )

    if args.show or args.save_figure is not None:
        fig = plot_optimization_comparison(df)
        if args.save_figure is not None:
            prefix = resolve_output_prefix(args.save_figure)
            saved_paths["figure_png"] = save_figure(
                fig,
                prefix.parent / f"{prefix.name}_compare.png",
            )
        if args.show:
            import matplotlib.pyplot as plt

            plt.show()
        else:
            import matplotlib.pyplot as plt

            plt.close(fig)

    for label, path in saved_paths.items():
        print(f"Saved {label}: {path}")
    return 0


def handle_optimize_sensitivity(args: argparse.Namespace) -> int:
    if not args.show:
        os.environ.setdefault("MPLBACKEND", "Agg")

    from optimize import sensitivity_analysis
    from optimize_viz import plot_optimization_sensitivity

    cfg = build_optimizer_config(args.config, args)
    df = sensitivity_analysis(
        config=cfg,
        K_max_values=args.K_max_values,
        n_od_samples=args.samples,
        demand_mode=args.demand_model,
        origin_weight_col=args.origin_weight_col,
        dest_weight_col=args.dest_weight_col,
        dest_jitter_m=args.dest_jitter_m,
        beta=args.beta,
        data_dir=str(args.data_dir),
        verbose=True,
    )
    print("\n" + df.to_string(index=False))

    saved_paths = {}
    if args.save_results is not None:
        prefix = resolve_output_prefix(args.save_results)
        saved_paths["sensitivity_csv"] = save_dataframe(
            df,
            prefix.parent / f"{prefix.name}_sensitivity.csv",
        )

    if args.show or args.save_figure is not None:
        topo, proto = args.config
        fig = plot_optimization_sensitivity(
            df,
            title=f"Sensitivity: {topo} / {proto}",
        )
        if args.save_figure is not None:
            prefix = resolve_output_prefix(args.save_figure)
            saved_paths["figure_png"] = save_figure(
                fig,
                prefix.parent / f"{prefix.name}_sensitivity.png",
            )
        if args.show:
            import matplotlib.pyplot as plt

            plt.show()
        else:
            import matplotlib.pyplot as plt

            plt.close(fig)

    for label, path in saved_paths.items():
        print(f"Saved {label}: {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)

    top_level_commands = {"run", "optimize", "-h", "--help"}
    if argv and argv[0] not in top_level_commands:
        argv = ["run", *argv]

    if not argv:
        parser.print_help()
        return 0

    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 1
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())

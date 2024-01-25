from argparse import Namespace

from hw1.argparser import parse_args
from hw1.simulation import find_lagrange_points, initialize_system
from hw1.visualize import plot_gravity, plot_potential


def main(args: Namespace):
    system = initialize_system(args.sep, args.mass_1, args.mass_2)
    lagrange_points = find_lagrange_points(
        system,
        tol=args.tol,
        max_iter=args.maxiter,
        xlim=args.xlim,
        ylim=args.ylim,
    )

    with open(f"lagrange_points_M{args.mass_1}_m{args.mass_2}.txt", "w") as f:
        for idx, point in enumerate(lagrange_points):
            f.write(f"L{idx+1} {point[0]} {point[1]}\n")

    plot_potential(
        system,
        grid_size=args.grid_size,
        xlim=args.xlim,
        ylim=args.ylim,
        filename=f"potential_M{args.mass_1}_m{args.mass_2}.pdf",
    )

    plot_gravity(
        system,
        lagrange_points,
        grid_size=args.grid_size,
        xlim=args.xlim,
        ylim=args.ylim,
        filename=f"vectorplot_M{args.mass_1}_m{args.mass_2}.pdf",
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)

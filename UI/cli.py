import click
from Logic.Simulation import run_simulation

@click.command()
@click.option('--n', default=10, help='An integer value for the number of bodies in the simulation. Default is 20 bodies')
@click.option('--R0', default=3.0856776e10, help='Initial radius of the cluster')
@click.option('--timespan', default=1e6, help='Time span in seconds for the simulation')
@click.option('--theta', default=0.5, help='Opening angle parameter for the Barnes-Hut algorithm')
@click.option('--bound-condition', default=10, help='Limit at which stars are no longer a part of the cluster')

def cli(n, R0, timespan, theta, bound_condition):
    """CLI for running the N-body simulation."""
    run_simulation(n, R0, timespan, theta, bound_condition)

if __name__ == '__main__':
    cli()
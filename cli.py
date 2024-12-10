
import click
from main import main_function

@click.command()
@click.option('--option', default='default', help='An example option.')
def cli(option):
    """CLI entry point."""
    main_function(option)

if __name__ == '__main__':
    cli()
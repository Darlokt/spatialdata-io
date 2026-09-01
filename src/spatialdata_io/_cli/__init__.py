"""Private Click application for spatialdata-io."""

from __future__ import annotations

from spatialdata_io._cli._group import cli
from spatialdata_io._cli.commands.codex import codex_command
from spatialdata_io._cli.commands.cosmx import cosmx_command
from spatialdata_io._cli.commands.curio import curio_command
from spatialdata_io._cli.commands.dbit import dbit_command
from spatialdata_io._cli.commands.generic import generic_command
from spatialdata_io._cli.commands.iss import iss_command
from spatialdata_io._cli.commands.macsima import macsima_command
from spatialdata_io._cli.commands.mcmicro import mcmicro_command
from spatialdata_io._cli.commands.merscope import merscope_command
from spatialdata_io._cli.commands.seqfish import seqfish_command
from spatialdata_io._cli.commands.steinbock import steinbock_command
from spatialdata_io._cli.commands.stereoseq import stereoseq_command
from spatialdata_io._cli.commands.visium import visium_command
from spatialdata_io._cli.commands.visium_hd import visium_hd_command
from spatialdata_io._cli.commands.xenium import xenium_command

cli.add_command(codex_command)
cli.add_command(cosmx_command)
cli.add_command(curio_command)
cli.add_command(dbit_command)
cli.add_command(generic_command)
cli.add_command(iss_command)
cli.add_command(macsima_command)
cli.add_command(mcmicro_command)
cli.add_command(merscope_command)
cli.add_command(seqfish_command)
cli.add_command(steinbock_command)
cli.add_command(stereoseq_command)
cli.add_command(visium_command)
cli.add_command(visium_hd_command)
cli.add_command(xenium_command)

__all__ = ["cli"]

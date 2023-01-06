import hdl21 as h
import hdl21.sim as hs
from hdl21 import Diff
from hdl21.pdk import Corner
from hdl21.prefix import MILLI
from hdl21.primitives import Vdc, Idc, C, Vpulse
from vlsirtools.spice import SimOptions, SupportedSimulators, ResultFormat

from . import scratch

sim_options = SimOptions(
    rundir=scratch,
    fmt=ResultFormat.SIM_DATA,
    simulator=SupportedSimulators.NGSPICE,
)


@h.paramclass
class Pvt:
    """Process, Voltage, and Temperature Parameters"""

    p = h.Param(dtype=Corner, desc="Process Corner", default=Corner.TYP)
    v = h.Param(dtype=h.Prefixed, desc="Supply Voltage Value (V)", default=1800 * MILLI)
    t = h.Param(dtype=int, desc="Simulation Temperature (C)", default=25)


@h.paramclass
class DiffClkParams:
    """Differential Clock Generator Parameters"""

    period = h.Param(dtype=hs.ParamVal, desc="Period")
    delay = h.Param(dtype=hs.ParamVal, desc="Delay")
    vd = h.Param(dtype=hs.ParamVal, desc="Differential Voltage")
    vc = h.Param(dtype=hs.ParamVal, desc="Common-Mode Voltage")
    trf = h.Param(dtype=hs.ParamVal, desc="Rise / Fall Time")


@h.generator
def DiffClkGen(p: DiffClkParams) -> h.Module:
    """# Differential Clock Generator
    For simulation, from ideal pulse voltage sources"""

    ckg = h.Module()
    ckg.VSS = VSS = h.Port()
    ckg.ck = ck = Diff(role=Diff.Roles.SINK, port=True)

    def vparams(polarity: bool) -> Vpulse.Params:
        """Closure to create the pulse-source parameters for each differential half.
        Argument `polarity` is True for positive half, False for negative half."""
        # Initially create the voltage levels for the positive half
        v1 = p.vc + p.vd / 2
        v2 = p.vc - p.vd / 2
        if not polarity:  # And for the negative half, swap them
            v1, v2 = v2, v1
        return Vpulse.Params(
            v1=v1,
            v2=v2,
            period=p.period,
            rise=p.trf,
            fall=p.trf,
            width=p.period / 2 - p.trf,
            delay=p.delay,
        )

    # Create the two complementary pulse-sources
    ckg.vp = Vpulse(vparams(True))(p=ck.p, n=VSS)
    ckg.vn = Vpulse(vparams(False))(p=ck.n, n=VSS)
    return ckg

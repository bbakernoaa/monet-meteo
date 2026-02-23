import numpy as np
from monet_meteo.thermodynamics.thermodynamic_calculations import (
    saturation_vapor_pressure,
)


def dewpoint_debug(temp, rh):
    es = saturation_vapor_pressure(temp)
    e_hpa = (rh * es) / 100.0
    a, b = 17.67, 243.5
    val = np.log(e_hpa / 6.112)
    td_c = b * val / (a - val)
    print(f"temp={temp}, rh={rh}, es={es}, e_hpa={e_hpa}, val={val}, td_c={td_c}")
    return td_c


dewpoint_debug(-20.0, 0.1)

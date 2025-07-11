from .common import *


def search_molecular_line(restfreq, unit="GHz", species_id=None, 
                          printinfo=True, return_table=False):
    """
    Search for molecular line information given a rest frequency.
    
    Parameters:
    - restfreq (float): The rest frequency of the molecular line.
    - unit (str, optional): The unit of the rest frequency. Default is "GHz".
    - printinfo (bool, optional): Whether to print the retrieved information. Default is True.
    
    Returns:
    - tuple: A tuple containing the following information about the molecular line:
        - species (str): The species of the molecular line.
        - chemical_name (str): The chemical name of the species.
        - freq (float): The frequency of the molecular line in GHz.
        - freq_err (float): The measurement error of the frequency in GHz.
        - qns (str): The resolved quantum numbers.
        - CDMS_JPL_intensity (float): The CDMS/JPL intensity.
        - Sijmu_sq (float): The S_ij * mu^2 value in Debye^2.
        - Sij (float): The S_ij value.
        - Aij (float): The Einstein A coefficient in 1/s.
        - Lovas_AST_intensity (float): The Lovas/AST intensity.
        - lerg (float): The lower energy level in Kelvin.
        - uerg (float): The upper energy level in Kelvin.
        - gu (float): The upper state degeneracy.
        - constants (tuple): The rotational constants (A, B, C) in MHz.
        - source (str): The source of the data.
    
    Raises:
    - ValueError: If the rest frequency is None.
    
    Notes:
    - This function requires an internet connection to query the Splatalogue database.
    - If the frequency does not closely match any known molecular lines, a warning will be printed.
    """
    from .utils import _best_match_line
    
    # error checking for rest frequency
    if restfreq is None:
        raise ValueError("The rest frequency cannot be 'None'.")
        
    if unit != "GHz":
        restfreq = u.Quantity(restfreq, unit).to_value(u.GHz)   # convert unit to GHz
    
    results = _best_match_line(restfreq, species_id=species_id, 
                               return_table=return_table)
    if return_table:
        return results

    # find information of the line 
    species_id = results["SPECIES_ID"]
    species = results["SPECIES"]
    chemical_name = results["CHEMICAL_NAME"]
    freq = results["FREQUENCY"]
    qns = results["QUANTUM_NUMBERS"]
    intensity = results["INTENSITY"]
    Sijmu_sq = results["SMU2"]
    log10_Aij = results["LOGA"]
    Aij = 10**log10_Aij
    lerg = results["EL"]
    uerg = results["EU"]
    try:
        upper_state, lower_state = map(int, qns.strip("J=").split('-'))
        gu = 2 * upper_state + 1
    except ValueError:
        warnings.warn("Failed to calculated upper state degeneracy.")
        gu = None
    source = results["LINELIST"]

    # find species id
    if species_id is None:
        warnings.warn("Failed to find species ID / rotational constants. \n")
        constants = None
        url = None
        display_url = None
    else:
        # find rotational constant
        url = f"https://splatalogue.online/splata-slap/species/{species_id}"
        display_url = f"https://splatalogue.online/#/species?id={species_id}"
        try:
            # search the web for rotational constants 
            from urllib.request import urlopen
            page = urlopen(url)
            html = page.read().decode("utf-8")
            metadata = eval(html.replace("null", "None"))[0]['metaData']  # list of dictionaries
        except:
            print(f"Failed to read webpage: {display_url} \n")
            print(f"Double check internet connection / installation of 'urllib' module.")
            url = None
            constants = None
        else:
            a_const = metadata.get("A")
            b_const = metadata.get("B")
            c_const = metadata.get("C")
            constants = tuple((float(rot_const) if rot_const is not None else None) \
                              for rot_const in (a_const, b_const, c_const))
                
    # store data in a list to be returned and convert masked data to NaN
    data = [species, chemical_name, freq, None, qns, intensity, Sijmu_sq,
            None, Aij, None, lerg, uerg, gu, constants, source]
    
    for i, item in enumerate(data):
        if np.ma.is_masked(item):
            data[i] = np.nan
    
    # print information if needed
    if printinfo:
        print(15*"#" + "Line Information" + 15*"#")
        print(f"Species ID: {species_id}")
        print(f"Species: {data[0]}")
        print(f"Chemical name: {data[1]}")
        print(f"Frequency: {data[2]} +/- {data[3]} [GHz]")
        print(f"Resolved QNs: {data[4]}")
        if not np.isnan(data[5]) and data[5] != 0:
            print(f"Intensity: {data[5]}")        
        print(f"Sij mu2: {data[6]} [Debye2]")
        print(f"Sij: {data[7]}")
        print(f"Einstein A coefficient (Aij): {data[8]:.3e} [1/s]")
        print(f"Lower energy level: {data[10]} [K]")
        print(f"Upper energy level: {data[11]} [K]")
        print(f"Upper state degeneracy (gu): {data[12]}")
        if data[13] is not None:
            print("Rotational constants:")
            if data[13][0] is not None:
                print(f"    A0 = {data[13][0]} [MHz]")
            if data[13][1] is not None:
                print(f"    B0 = {data[13][1]} [MHz]")
            if data[13][2] is not None:
                print(f"    C0 = {data[13][2]} [MHz]")
        print(f"Source: {data[14]}")
        print(46*"#")
        if url is not None:
            print(f"Link to species data: {display_url} \n")
    
    # return data
    return tuple(data)


def planck_function(v, T):
    """
    Public function to calculate the planck function value.
    Parameters:
        v (float): frequency of source [GHz]
        T (float): temperature [K]
    Returns:
        Planck function value [Jy]
    """
    # constants
    h = const.h.cgs
    clight = const.c.cgs
    k = const.k_B.cgs
    
    # assign units
    if not isinstance(v, u.Quantity):
        v *= u.GHz
    if not isinstance(T, u.Quantity):
        T *= u.K
    
    # calculate
    Bv = 2*h*v**3/clight**2 / (np.exp(h*v/k/T)-1)
    
    # return value
    return Bv.to_value(u.Jy)


def H2_column_density(continuum, T_dust, k_v):
    """
    Public function to calculate the H2 column density from continuum data.
    Parameters:
        continuum (Spatialmap): the continuum map
        T_dust (float): dust temperature [K]
        k_v (float): dust-mass opacity index [cm^2/g]
    Returns:
        H2_cd (Spatialmap): the H2 column density map [cm^-2]
    """
    # constants
    m_p = const.m_p.cgs  # proton mass
    mmw = 2.8            # mean molecular weight
    
    # convert continuum to brightness temperature
    I_v = continuum.conv_bunit("Jy/sr", inplace=False)
    if not isinstance(I_v.data, u.Quantity):
        I_v *= u.Jy
    v = continuum.restfreq * u.Hz
    
    # assign units:
    if not isinstance(T_dust, u.Quantity):
        T_dust *= u.K
    if not isinstance(k_v, u.Quantity):
        k_v *= u.cm**2 / u.g
    
    # start calculating
    B_v = planck_function(v=v.to_value(u.GHz), T=T_dust)*u.Jy
    H2_cd = (I_v / (k_v*B_v*mmw*m_p)).to_value(u.cm**-2)

    return H2_cd


def J_v(v, T):
    """
    Public function to calculate the Rayleigh-Jeans equivalent temperature.
    Parameters:
        v (float): frequency (GHz)
        T (float): temperature (K)
    Returns:
        Jv (float): Rayleigh-Jeans equivalent temperature (K)
    """
    # constants
    k = const.k_B.cgs
    h = const.h.cgs
    
    # assign unit
    if not isinstance(v, u.Quantity):
        v *= u.GHz
    if not isinstance(T, u.Quantity):
        T *= u.K
    
    # calculate R-J equivalent temperature
    Jv = (h*v/k) / (np.exp(h*v/k/T)-1)

    return Jv.to(u.K)


def column_density_linear_optically_thin(image, T_ex, T_bg=2.726, B0=None, R_i=1, f=1.):
    """
    Public function to calculate the column density of a linear molecule using optically thin assumption.
    Source: https://doi.org/10.48550/arXiv.1501.01703
    Parameters:
        image (Spatialmap/Datacube): the moment 0 / datacube map
        T_ex (float): the excitation temperature [K]
        T_bg (float): background temperature [K]. 
                      Default is to use cosmic microwave background (2.726 K).
        R_i (float): Relative intensity of transition. 
                     Default is to consider simplest case (= 1)
        f (float): a correction factor accounting for source area being smaller than beam area.
                   Default = 1 assumes source area is larger than beam area.
    Returns:
        cd_img (Spatialmap/Datacube): the column density map
    """
    from .spatialmap import Spatialmap
    from .datacube import Datacube
    
    # constants
    k = const.k_B.cgs
    h = const.h.cgs
    
    # assign units
    if not isinstance(T_ex, u.Quantity):
        T_ex *= u.K
    if not isinstance(T_bg, u.Quantity):
        T_bg *= u.K
    
    # convert units 
    if isinstance(image, Spatialmap):
        image = image.conv_bunit("K.km/s", inplace=False)*u.K*u.km/u.s
    elif isinstance(image, Datacube):
        image = image.conv_bunit("K", inplace=False)
        image = image.conv_specunit("km/s", inplace=False)
        dv = image.header["dv"]
        image = image*dv*(u.K*u.km/u.s)
    else:
        raise ValueError(f"Invalid data type for image: {type(image)}")
    
    # get info
    line_data = image.line_info(printinfo=True)
    v = line_data[2]*u.GHz  # rest frequency
    S_mu2 = line_data[6]*(1e-18**2)*(u.cm**5*u.g/u.s**2)  # S mu^2 * g_i*g_j*g_k [debye2]
    E_u = line_data[11]*u.K  # upper energy level 
    if B0 is None:
        B0 = line_data[13][1]*u.MHz  # rotational constant
    elif not isinstance(B0, u.Quantity):
        B0 *= u.MHz
    Q_rot = _Qrot_linear(T=T_ex, B0=B0)  # partition function
        
    # error checking to make sure molecule is linear
    if line_data[13] is not None:
        if line_data[13][0] is not None or line_data[13][2] is not None:
            raise Exception("The molecule is not linear.")

    # calculate column density
    aa = 3*h/(8*np.pi**3*S_mu2*R_i)
    bb = Q_rot
    cc = np.exp(E_u/T_ex) / (np.exp(h*v/k/T_ex)-1)
    dd = 1 / (J_v(v=v, T=T_ex)-J_v(v=v, T=T_bg))
    constant = aa*bb*cc*dd/f
    cd_img = constant*image
    cd_img = cd_img.to_value(u.cm**-2)

    return cd_img


def column_density_linear_optically_thick(moment0_map, T_ex, tau, T_bg=2.726, R_i=1, f=1):
    """
    Function to calculate the column density of a linear molecule using optically thick assumption.
    Source: https://doi.org/10.48550/arXiv.1501.01703
    Parameters:
        moment0_map (Spatialmap): the moment 0 map
        T_ex (float): the excitation temperature [K]
        tau (float): gas optical depth
        T_bg (float): background temperature [K]. 
                      Default is to use cosmic microwave background (2.726 K).
        R_i (float): Relative intensity of transition. 
                     Default is to consider simplest case (= 1)
        f (float): a correction factor accounting for source area being smaller than beam area.
                   Default = 1 assumes source area is larger than beam area.
    Returns:
        cd_img (Spatialmap): the column density map
    """
    # calculate using optically thin assumption
    cd_opt_thin = column_density_linear_optically_thin(moment0_map=moment0_map,
                                                       T_ex=T_ex,
                                                       T_bg=T_bg,
                                                       R_i=R_i,
                                                       f=f)
    # correction factor, relates optically thin to optically thick case
    corr_factor = tau/(1-np.exp(-tau))
    cd_img = corr_factor*cd_opt_thin
    return cd_img
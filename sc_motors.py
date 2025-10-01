def kv_to_rpm(kv: float, volt: float) -> float:
    """
    Calculate RPM from kV and voltage.
    
    Args:
        kv (float): The motor's kV rating.
        volt (float): The input voltage.
    
    Returns:
        float: Estimated unloaded RPM.
    """
    return kv * volt

def rpm_to_kv(rpm: float, volt: float) -> float:
    """
    Calculate kV from RPM and voltage.
    
    Args:
        rpm (float): The motor's RPM.
        volt (float): The input voltage.
    
    Returns:
        float: Estimated kV rating.
    """
    return rpm / volt

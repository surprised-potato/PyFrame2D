# units/units.py

import re
import math
# Removed Decimal as float is sufficient and simpler here

# --- Constants ---

# Dictionary of SI prefixes and their multipliers
PREFIXES = {
    'Y': 1e24, 'Z': 1e21, 'E': 1e18, 'P': 1e15, 'T': 1e12,
    'G': 1e9,  'M': 1e6,  'k': 1e3,  'h': 1e2,  'da': 1e1,
    '': 1.0,   # Base unit (no prefix)
    'd': 1e-1, 'c': 1e-2, 'm': 1e-3, 'µ': 1e-6, 'u': 1e-6, # Allow 'u' for micro
    'n': 1e-9, 'p': 1e-12, 'f': 1e-15, 'a': 1e-18, 'z': 1e-21, 'y': 1e-24
}

# Prefixes preferred for formatting (exclude less common ones like hecto, deca, deci, centi)
FORMATTING_PREFIXES = {
    'Y': 1e24, 'Z': 1e21, 'E': 1e18, 'P': 1e15, 'T': 1e12,
    'G': 1e9,  'M': 1e6,  'k': 1e3,
    '': 1.0,   # Base unit
    'm': 1e-3, 'µ': 1e-6, 'u': 1e-6,
    'n': 1e-9, 'p': 1e-12, 'f': 1e-15, 'a': 1e-18, 'z': 1e-21, 'y': 1e-24
}

# Reverse mapping and sorted list for formatting using preferred prefixes
FORMATTING_PREFIX_SYMBOLS = {v: k for k, v in FORMATTING_PREFIXES.items() if k != 'u'} # Prefer µ over u for output
FORMATTING_PREFIX_SYMBOLS.update({1e-6: 'µ'}) # Ensure µ is the symbol for 1e-6
# Sort multipliers from largest to smallest for selection
FORMATTING_SORTED_MULTIPLIERS = sorted(FORMATTING_PREFIX_SYMBOLS.keys(), reverse=True)


# Define the base units expected in the structural analysis context
BASE_UNITS = {
    "N": "Force",          # Newton
    "m": "Length",         # Meter
    "Pa": "Pressure/Stress",# Pascal (N/m^2)
    "rad": "Angle"         # Radian
}

# --- Parsing Function ---

def parse_value_unit(input_str: str, expected_base_unit: str) -> float:
    """
    Parses a string containing a value and a unit (with optional SI prefix)
    and converts it to the corresponding base SI unit value.

    Args:
        input_str: The string to parse (e.g., "10 kN", "500mm", "200 GPa").
                   Allows optional space between value and unit.
        expected_base_unit: The target base SI unit ("N", "m", "Pa", "rad").

    Returns:
        The numeric value converted to the expected base SI unit (float).

    Raises:
        ValueError: If the input string format is invalid, the unit is unknown,
                    the prefix is unknown, or the unit doesn't match the
                    expected base unit type.
        TypeError: If input_str is not a string or expected_base_unit is invalid.
    """
    if not isinstance(input_str, str):
        raise TypeError("Input must be a string.")
    if expected_base_unit not in BASE_UNITS:
        raise TypeError(f"Invalid expected_base_unit: '{expected_base_unit}'. Must be one of {list(BASE_UNITS.keys())}")

    input_str = input_str.strip()

    # Regex: Allows optional space between number and unit. Handles 'u' or 'µ'.
    # Allows numbers like "10", ".5", "1.2e3", "-3.4E-2"
    match = re.match(r"^\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*([a-zA-Zµμ]+)\s*$", input_str)

    if not match:
        # Try checking if it's just a number (treat as base unit)
        try:
            value = float(input_str)
            # If it looks like just a number, assume it's in the base unit
            # This is a convenience, might need stricter validation depending on requirements
            print(f"Warning: Input '{input_str}' contains only a number. Assuming base unit '{expected_base_unit}'.")
            return value
        except ValueError:
             # If it's not a plain number either, then the format is truly invalid
             raise ValueError(f"Invalid format: '{input_str}'. Expected format like '10 kN', '500mm', '-1.5e3 Pa'.")


    numeric_str, unit_str = match.groups()

    # Convert numeric part to float
    try:
        value = float(numeric_str)
    except ValueError:
        # This path is less likely now due to regex, but good safeguard
        raise ValueError(f"Invalid numeric value extracted: '{numeric_str}' from input '{input_str}'")

    # Identify prefix and base unit
    parsed_prefix = ""
    parsed_base_unit = ""

    # Check if the unit_str directly matches the expected base unit
    if unit_str == expected_base_unit:
        parsed_prefix = ""
        parsed_base_unit = expected_base_unit
    else:
        # If not a direct match, check for known prefixes combined with the expected base unit
        found_match = False
        # Iterate prefixes from longest to shortest to avoid partial matches (e.g., 'da' vs 'd')
        for p in sorted(PREFIXES.keys(), key=len, reverse=True):
            if not p: continue # Skip the empty base prefix here
            if unit_str.startswith(p):
                potential_base = unit_str[len(p):]
                if potential_base == expected_base_unit:
                    # Check if this prefix is valid
                    if p in PREFIXES:
                        parsed_prefix = p
                        parsed_base_unit = potential_base
                        found_match = True
                        break # Found the correct prefix and base unit combination
                    else:
                        # Prefix matched but it's not one we recognize (shouldn't happen with current PREFIXES dict)
                         raise ValueError(f"Internal error: Matched unknown prefix '{p}' in '{unit_str}'")

        if not found_match:
            # If loop finishes without a match, the unit is unknown or mismatched
            raise ValueError(f"Unit mismatch or unknown unit/prefix combination: '{unit_str}'. Expected base unit: '{expected_base_unit}' or a prefixed version.")

    # Get the multiplier for the identified prefix
    multiplier = PREFIXES.get(parsed_prefix, 1.0) # Default to 1.0 if prefix is ""

    # Calculate the value in the base SI unit
    base_value = value * multiplier

    return base_value

# --- Formatting Function ---

def format_value_unit(value: float, base_unit_str: str, precision: int = 3) -> str: # Default precision 3
    """
    Formats a numeric value (assumed to be in base SI units) into a
    human-readable string with an appropriate SI prefix (preferring common ones).

    Args:
        value: The numeric value in base SI units (e.g., 15000.0 for 15 kN).
        base_unit_str: The base SI unit ("N", "m", "Pa", "rad").
        precision: The number of significant figures for formatting.

    Returns:
        A formatted string (e.g., "15.0 kN", "500 mm", "2.00 GPa").

    Raises:
        TypeError: If value is not numeric or base_unit_str is invalid.
    """
    if not isinstance(value, (int, float)):
        raise TypeError("Input value must be numeric.")
    if base_unit_str not in BASE_UNITS:
        raise TypeError(f"Invalid base_unit_str: '{base_unit_str}'. Must be one of {list(BASE_UNITS.keys())}")

    if value == 0:
        return f"0 {base_unit_str}" # Consistent zero representation

    abs_value = abs(value)
    best_multiplier = 1.0
    best_prefix = ''

    # Try large prefixes first (k, M, G, T, ...) -> largest to smallest
    for m in FORMATTING_SORTED_MULTIPLIERS:
        if m <= 1.0: continue # Only consider multipliers > 1.0
        if abs_value >= m:
            best_multiplier = m
            best_prefix = FORMATTING_PREFIX_SYMBOLS[m]
            break # Found the largest suitable prefix > 1.0

    # If no large prefix was suitable (multiplier is still 1.0) AND value < 1.0
    # try small prefixes (m, µ, n, ...) -> largest to smallest (e.g., m before µ)
    if best_multiplier == 1.0 and abs_value < 1.0:
        # Find the largest multiplier M < 1.0 such that abs_value / M >= 1.0
        # Iterate prefixes < 1.0, from largest (milli) down
        for m in FORMATTING_SORTED_MULTIPLIERS:
             if m >= 1.0: continue # Skip base and large prefixes
             # Check if using this prefix results in a value >= 1.0
             # Added a small tolerance (1e-9) for floating point comparisons
             if abs_value / m >= (1.0 - 1e-9):
                 best_multiplier = m
                 best_prefix = FORMATTING_PREFIX_SYMBOLS[m]
                 break # Found the best small prefix

    # If best_multiplier is still 1.0, it means:
    # - Value is >= 1.0 but less than the smallest large prefix (kilo)
    # - Or value is < 1.0 but also smaller than the smallest prefix that would make it >= 1.0

    scaled_value = value / best_multiplier

    # Format the scaled value - Use '.<precision>G'
    # Ensure sufficient precision to avoid unwanted scientific notation for moderate numbers.
    format_spec = f".{precision}G"
    formatted_value = f"{scaled_value:{format_spec}}"

    # Handle potential 'u' vs 'µ' preference - always output 'µ'
    display_prefix = 'µ' if best_prefix == 'u' else best_prefix

    return f"{formatted_value} {display_prefix}{base_unit_str}"


# --- Example Usage (Optional - can be run directly) ---
if __name__ == "__main__":
    print("--- Parsing Examples ---")
    inputs = ["10 kN", "500 mm", "500mm", "2 GPa", "1.5 rad", "-2.5 MN", "0.1 m", "1500 µm", "1.2e3 N", " 100Pa ", ".5 m", "-.2kN", "10"]
    expected_units = ["N", "m", "m", "Pa", "rad", "N", "m", "m", "N", "Pa", "m", "N", "N"]

    for instr, exp_unit in zip(inputs, expected_units):
        try:
            base_val = parse_value_unit(instr, exp_unit)
            print(f"'{instr}' ({exp_unit}) -> {base_val:.6g} {exp_unit}")
        except (ValueError, TypeError) as e:
            print(f"Error parsing '{instr}': {e}")

    print("\n--- Formatting Examples ---")
    values = [15000.0, 0.5, 2e9, 1.5, -2.5e6, 0.0015, 1234567, 0.0, 1.23, 0.9, -1500.0, 1.23e-7]
    units = ["N", "m", "Pa", "rad", "N", "m", "N", "Pa", "m", "N", "Pa", "m"]

    for val, unit in zip(values, units):
        try:
            formatted_str = format_value_unit(val, unit, precision=3) # Use prec=3 for demo
            print(f"{val:.6g} {unit} -> '{formatted_str}'")
        except TypeError as e:
            print(f"Error formatting {val} {unit}: {e}")

    print("\n--- Error Handling Examples ---")
    error_inputs = [("10 k N", "N"), ("abc N", "N"), ("10 xyz", "N"), ("10 kN", "m"), (100, "N"), ("--10 N", "N"), ("10.. N", "N"), ("10. N", "N")]
    for instr, exp_unit in error_inputs:
         try:
             parse_value_unit(instr, exp_unit)
         except (ValueError, TypeError) as e:
             print(f"Correctly caught error for '{instr}' ({exp_unit}): {e}")
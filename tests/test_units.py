# tests/test_units.py
import re
import pytest # type: ignore
from units.units import parse_value_unit, format_value_unit

# Using pytest.approx for floating point comparisons
from pytest import approx # type: ignore

# --- Tests for parse_value_unit ---

@pytest.mark.parametrize("input_str, expected_unit, expected_value", [
    # Basic SI units
    ("10 N", "N", 10.0),
    ("5 m", "m", 5.0),
    ("200 Pa", "Pa", 200.0),
    ("1.5 rad", "rad", 1.5),
    ("0 N", "N", 0.0),
    # Prefixed units
    ("10 kN", "N", 10000.0),
    ("500 mm", "m", 0.5),
    ("2 GPa", "Pa", 2e9),
    ("1500 µm", "m", 0.0015),
    ("1500 um", "m", 0.0015), # Allow 'u' for micro
    ("0.1 MPa", "Pa", 100000.0),
    ("2.5 cm", "m", 0.025), # Should still parse 'c' even if not preferred for formatting
    ("1 daN", "N", 10.0), # Should parse 'da'
    ("2 hm", "m", 200.0), # Should parse 'h'
    ("3 dN", "N", 0.3),   # Should parse 'd'
    # Negative values
    ("-2.5 MN", "N", -2.5e6),
    ("-100 mm", "m", -0.1),
    # Scientific notation
    ("1.2e3 kN", "N", 1.2e6),
    ("3.4E-2 mm", "m", 3.4e-5),
    ("-2e-3 GPa", "Pa", -2e6),
    # Whitespace handling
    (" 100 N ", "N", 100.0),
    ("  2.5kPa ", "Pa", 2500.0), # No space needed now
    ("\t50mm", "m", 0.05),      # No space needed now
    # Decimal values without leading zero
    (".5 m", "m", 0.5),
    ("-.2 kN", "N", -200.0),
    # No space between value and unit
    ("10kN", "N", 10000.0),
    ("500mm", "m", 0.5),
    ("2GPa", "Pa", 2e9),
    ("-2.5MN", "N", -2.5e6),
    (".5m", "m", 0.5),
    # Plain numbers (should issue warning and assume base unit)
    ("150", "N", 150.0),
    ("-20.5", "m", -20.5),
    ("1e6", "Pa", 1e6),
    #conversion tests
    ("1000 mm", "m", 1.0),       # mm to m
    ("5 kN", "N", 5000.0),       # kN to N
    ("2.5 GPa", "Pa", 2.5e9)

])
def test_parse_value_unit_valid(input_str, expected_unit, expected_value, capsys):
    """Tests successful parsing of various valid input strings."""
    # capsys fixture captures print output (like warnings)
    assert parse_value_unit(input_str, expected_unit) == approx(expected_value)
    # Check if a warning was printed for plain number inputs
    captured = capsys.readouterr()
    is_plain_number = False
    try:
        float(input_str)
        is_plain_number = True
    except ValueError:
        pass
    if is_plain_number and not re.match(r"^\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*([a-zA-Zµμ]+)\s*$", input_str.strip()):
         assert "Warning: Input" in captured.out
    else:
         assert "Warning: Input" not in captured.out


@pytest.mark.parametrize("input_str, expected_unit, error_type, error_message_part", [
    # Invalid format (rejected by regex or float conversion)
    ("abc N", "N", ValueError, "Invalid format"),
    #("10", "N", ValueError, "Invalid format"), # Fails because unit missing now requires warning/special handling tested above
    ("10 k N", "N", ValueError, "Invalid format"), # Space within unit part
    ("--10 N", "N", ValueError, "Invalid format"), # Double minus sign
    ("10.. N", "N", ValueError, "Invalid format"), # Double decimal point
    ("10. N", "N", ValueError, "Invalid format"), # Decimal point at end without digits
    ("10 N m", "N", ValueError, "Invalid format"), # Multiple units
    # Unknown unit/prefix combination
    ("10 kg", "N", ValueError, "Unit mismatch or unknown unit/prefix combination"),
    ("5 meters", "m", ValueError, "Unit mismatch or unknown unit/prefix combination"),
    ("10 xN", "N", ValueError, "Unit mismatch or unknown unit/prefix combination"), # Unknown prefix 'x'
    ("10 k", "N", ValueError, "Unit mismatch or unknown unit/prefix combination"), # Prefix only
    # Correct prefix but mismatched expected base unit
    ("10 kN", "m", ValueError, "Unit mismatch or unknown unit/prefix combination"),
    ("500 mm", "Pa", ValueError, "Unit mismatch or unknown unit/prefix combination"),
    # Invalid expected unit type in function call
    ("10 N", "kg", TypeError, "Invalid expected_base_unit"),
])
def test_parse_value_unit_invalid(input_str, expected_unit, error_type, error_message_part):
    """Tests that parse_value_unit raises specific errors for invalid inputs."""
    with pytest.raises(error_type, match=error_message_part):
        parse_value_unit(input_str, expected_unit)

def test_parse_value_unit_invalid_input_type():
    """Tests that parse_value_unit raises TypeError for non-string input."""
    with pytest.raises(TypeError, match="Input must be a string"):
        parse_value_unit(100, "N")


# --- Tests for format_value_unit ---

@pytest.mark.parametrize("value, base_unit, precision, expected_str", [
    # Base cases (precision 4 default, adjusted where needed)
    (10.0, "N", 3, "10 N"),
    (5.0, "m", 3, "5 m"),
    (200.0, "Pa", 3, "200 Pa"),
    (1.5, "rad", 3, "1.5 rad"),
    (0.0, "N", 3, "0 N"),
    # Prefixed outputs (positive) - using preferred formatting prefixes
    (15000.0, "N", 3, "15 kN"),       # Prec 3 for cleaner output
    (15490.0, "N", 3, "15.5 kN"),     # Prec 3
    (1234567.0, "N", 3, "1.23 MN"),   # Prec 3
    (2e9, "Pa", 3, "2 GPa"),         # Prec 3
    (2.15e9, "Pa", 3, "2.15 GPa"),   # Prec 3
    (0.5, "m", 3, "500 mm"),         # No 'd', uses 'm' (milli)
    (0.025, "m", 2, "25 mm"),        # No 'c', uses 'm'
    (0.0015, "m", 2, "1.5 mm"),
    (0.00154, "m", 3, "1.54 mm"),     # Prec 3
    (0.0000015, "m", 2, "1.5 µm"),
    (1.23e-7, "m", 3, "123 nm"),      # Prec 3 needed for '123'
    # Prefixed outputs (negative)
    (-2.5e6, "N", 2, "-2.5 MN"),
    (-0.1, "m", 3, "-100 mm"),       # No 'd', uses 'm'
    (-1500.0, "Pa", 3, "-1.5 kPa"),
    # Values between 1.0 and smallest prefix
    (1.23, "m", 3, "1.23 m"),
    (0.9, "N", 3, "900 mN"),         # No 'd', uses 'm'
    (0.0009, "N", 3, "900 µN"),      # Uses 'µ', need precision 3
    # Very large/small values
    (5e13, "N", 2, "50 TN"),
    (3e-11, "m", 2, "30 pm"),
    # Edge case near 1 for prefix switch
    (999.0, "N", 3, "999 N"),
    (1000.0, "N", 3, "1 kN"),
    (0.00099, "m", 3, "990 µm"),     # Need precision 3
    (0.001, "m", 2, "1 mm"),

])
def test_format_value_unit_valid(value, base_unit, precision, expected_str):
    """Tests successful formatting of values into strings with units."""
    assert format_value_unit(value, base_unit, precision) == expected_str

@pytest.mark.parametrize("value, base_unit", [
    ("not a number", "N"),
    (100.0, "kg"),
])
def test_format_value_unit_invalid_input_type(value, base_unit):
    """Tests that format_value_unit raises TypeError for invalid input types."""
    with pytest.raises(TypeError):
        format_value_unit(value, base_unit)
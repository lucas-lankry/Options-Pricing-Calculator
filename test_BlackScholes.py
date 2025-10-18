import pytest
from BlackScholes import calculate_d1, calculate_d2, black_scholes

def main():
    test_calculate_d1()
    test_black_scholes()
    test_calculate_d2()

def test_black_scholes():
    with pytest.raises(ValueError):
        black_scholes(120,100,1.5,0.45,0.12,0)
    with pytest.raises(ValueError):
        black_scholes(120,100,1.5,0.45,0.12,"dog")

def test_calculate_d1():
    with pytest.raises(ValueError):
        calculate_d1(120, 100, 0, 0.45, 12)
    with pytest.raises(ValueError):
        calculate_d1(120, 0, 1.5, 0.45, 12)
    with pytest.raises(ValueError):
        calculate_d1(120, 100, 1.5, 0.45, 0)

def test_calculate_d2():
    with pytest.raises(ValueError):
        calculate_d2(None, 1.5, 12)

if __name__ == "__main__":
    main()
    

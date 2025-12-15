"""
Test EquiformerV2AdsorptionOracle on real adsorption systems
This script tests whether the model correctly predicts adsorption energies
"""

from ase.build import fcc111, molecule, add_adsorbate
from ase import Atoms
from chem_gym.surrogate.ocp_model_adsorption import EquiformerV2AdsorptionOracle
import numpy as np


def create_adsorbed_system(surface_element='Cu', size=(3, 3, 3),
                           adsorbate='CO', height=2.0):
    """Create a surface with adsorbate using proper ASE methods"""
    # Create surface
    surface = fcc111(surface_element, size=size, vacuum=10.0)
    surface.center()  # Center the slab

    # Create adsorbate
    ads = molecule(adsorbate)

    # Add adsorbate to surface at ontop site
    # Position: above first surface atom
    add_adsorbate(surface, ads, height, position=(0, 0))

    # Set tags to distinguish surface (1) from adsorbate (2)
    tags = np.ones(len(surface), dtype=int)
    # Last len(ads) atoms are the adsorbate
    tags[-len(ads):] = 2
    surface.set_tags(tags)

    return surface


def test_system(oracle, surface_element, adsorbate, size=(3, 3, 3)):
    """Test a specific surface+adsorbate system"""
    print(f"\n{'='*60}")
    print(f"Testing: {adsorbate} on {surface_element}({size[0]}x{size[1]})")
    print(f"{'='*60}")

    # Create system
    system = create_adsorbed_system(surface_element, size, adsorbate, height=2.0)

    n_total = len(system)
    n_surface = np.sum(system.get_tags() == 1)
    n_adsorbate = np.sum(system.get_tags() == 2)

    print(f"Total atoms: {n_total}")
    print(f"Surface atoms (tag=1): {n_surface}")
    print(f"Adsorbate atoms (tag=2): {n_adsorbate}")
    print(f"All tags: {system.get_tags()}")

    # Predict energies
    adsorption_energy, info = oracle.predict_adsorption_energy(system)

    print(f"\n=== Energy Breakdown ===")
    print(f"E_total (surface+adsorbate): {info['e_total']:.4f} eV")
    print(f"E_surface (clean surface):   {info['e_surface']:.4f} eV")
    print(f"E_adsorbate (gas phase):     {info['e_adsorbate']:.4f} eV")
    print(f"Adsorption energy:           {adsorption_energy:.4f} eV")

    # Physical expectations
    print(f"\n=== Physical Validation ===")
    print(f"Expected range for {adsorbate} on {surface_element}:")
    if adsorbate in ['CO', 'OH']:
        print("  Expected: -1.0 to -3.0 eV (chemisorption)")
        print("  physisorption would be: -0.1 to -0.5 eV")
    else:
        print("  Check literature for expected range")

    # Check if prediction is reasonable
    is_reasonable = adsorption_energy < -0.5
    status = "✓ REASONABLE" if is_reasonable else "✗ UNREASONABLE"
    print(f"\nPrediction: {status}")

    return info


def main():
    """Main test function"""
    print("Testing EquiformerV2AdsorptionOracle")
    print("="*60)

    # Load model
    checkpoint = "checkpoints/eq2_83M_2M.pt"
    oracle = EquiformerV2AdsorptionOracle(checkpoint, device="cuda")

    # Test different systems
    test_cases = [
        ('Cu', 'CO'),
        ('Cu', 'OH'),
        ('Cu', 'H'),
        ('Pt', 'CO'),  # Try different metal
    ]

    results = {}

    for metal, adsorbate in test_cases:
        try:
            info = test_system(oracle, metal, adsorbate)
            results[f"{adsorbate}_{metal}"] = info
        except Exception as e:
            print(f"ERROR testing {adsorbate} on {metal}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY OF ALL TESTS")
    print(f"{'='*60}")

    for key, info in results.items():
        print(f"{key:15s}: ΔE = {info['adsorption_energy']:8.4f} eV")

    print("\nExpected physical ranges:")
    print("  - Strong chemisorption: -1.5 to -3.5 eV")
    print("  - Weak chemisorption:   -0.5 to -1.5 eV")
    print("  - Physisorption:        -0.1 to -0.5 eV")
    print("  - Unstable:             > 0 eV")


if __name__ == "__main__":
    main()

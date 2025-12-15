"""
Correct adsorption energy test with proper structure creation
Avoids singular matrix errors by using proper ASE methods
"""

from ase.build import fcc111, molecule
from ase import Atoms
from ase.visualize import view
from chem_gym.surrogate.ocp_model import EquiformerV2Oracle
import numpy as np


def create_adsorption_system(element='Cu', size=(3, 3, 3), adsorbate='OH', height=2.0):
    """Create surface+adsorbate system with correct tags"""

    # Create surface
    surface = fcc111(element, size=size, vacuum=8.0)
    surface.center()

    # Ensure good periodic cell
    surface.set_pbc([True, True, False])

    # Create adsorbate
    ads = molecule(adsorbate)
    ads.center()

    # Create combined system
    combined = surface.copy()

    # Position adsorbate above ontop site (first surface atom)
    top_atom_pos = surface[0].position
    ads.translate(top_atom_pos + np.array([0, 0, height]))

    # Add adsorbate atoms to combined system
    for atom in ads:
        combined.append(atom)

    # Set tags: 1=surface atoms, 2=adsorbate atoms
    tags = np.ones(len(combined), dtype=int)
    tags[-len(ads):] = 2
    combined.set_tags(tags)

    return combined, surface, ads


def test_adsorption():
    """Main test function"""
    print("="*70)
    print("CORRECT ADSORPTION ENERGY TEST")
    print("="*70)

    # Load model
    checkpoint = "checkpoints/eq2_83M_2M.pt"
    oracle = EquiformerV2Oracle(checkpoint, device="cuda")

    # Create systems
    print("\n1. Creating systems...")
    adsorbed, surface, oh = create_adsorption_system(
        element='Cu',
        size=(3, 3, 3),
        adsorbate='OH',
        height=2.0
    )

    print(f"   Clean surface: {len(surface)} atoms")
    print(f"   OH adsorbate: {len(oh)} atoms")
    print(f"   Total system: {len(adsorbed)} atoms")
    print(f"   Tags: {np.unique(adsorbed.get_tags(), return_counts=True)}")

    # Verify structure integrity
    print("\n2. Verifying structure integrity...")
    print(f"   Cell: {adsorbed.cell.lengths()}")
    print(f"   PBC: {adsorbed.pbc}")

    cell_det = np.linalg.det(adsorbed.cell[:])
    print(f"   Cell determinant: {cell_det:.6f}")

    if abs(cell_det) < 1e-6:
        print("   ERROR: Singular cell matrix!")
        return
    else:
        print("   ✓ Cell matrix OK")

    # Predict energies
    print("\n3. Predicting energies...")
    print("-"*70)

    try:
        e_total = oracle.predict_energy(adsorbed)
        e_surface = oracle.predict_energy(surface)
        # Fix gas-phase molecule: add vacuum box
        oh_gas = oh.copy()
        oh_gas.set_cell([15.0, 15.0, 15.0])
        oh_gas.center()
        e_ads = oracle.predict_energy(oh_gas)

        print(f"E_total (surface + OH):     {e_total:12.4f} eV")
        print(f"E_surface (clean surface):  {e_surface:12.4f} eV")
        print(f"E_adsorbate (OH molecule):  {e_ads:12.4f} eV")
        print("-"*70)

        # Calculate adsorption energy
        adsorption_energy = e_total - e_surface - e_ads

        print(f"\n4. Adsorption energy calculation:")
        print(f"ΔE_ads = E_total - E_surface - E_adsorbate")
        print(f"ΔE_ads = {e_total:.4f} - {e_surface:.4f} - {e_ads:.4f}")
        print(f"ΔE_ads = {adsorption_energy:12.4f} eV")
        print("-"*70)

        # Physical interpretation
        print(f"\n5. Physical interpretation:")
        if adsorption_energy < -2.0:
            print(f"   ✓ STRONG CHEMISORPTION (ΔE = {adsorption_energy:.2f} eV)")
        elif adsorption_energy < -0.5:
            print(f"   ✓ WEAK CHEMISORPTION (ΔE = {adsorption_energy:.2f} eV)")
        elif adsorption_energy < 0:
            print(f"   ~ PHYSISORPTION (ΔE = {adsorption_energy:.2f} eV)")
        else:
            print(f"   ✗ UNSTABLE (ΔE = {adsorption_energy:.2f} eV)")

        print("\nExpected for OH on Cu(111):")
        print("   DFT predicts: -2.5 to -3.5 eV (strong chemisorption)")

    except Exception as e:
        print(f"\nERROR during prediction: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test with different heights
    print("\n" + "="*70)
    print("6. Testing height dependence...")
    print("="*70)

    heights = [1.5, 2.0, 2.5, 3.0, 4.0]
    energies = []

    for h in heights:
        ads_h, _, _ = create_adsorption_system(height=h)
        e_h = oracle.predict_energy(ads_h)
        e_ads_h = e_h - e_surface - e_ads
        energies.append(e_ads_h)
        print(f"Height {h:4.1f} Å: ΔE = {e_ads_h:8.4f} eV")

    # Check trend
    if len(energies) > 2:
        # Should be most stable at optimal binding distance
        min_energy = min(energies)
        min_height = heights[energies.index(min_energy)]
        print(f"\nMost stable at: {min_height:.1f} Å (ΔE = {min_energy:.4f} eV)")

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)


if __name__ == "__main__":
    test_adsorption()

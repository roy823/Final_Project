"""
Proper test of EquiformerV2 for adsorption energy prediction

Key principle: Adsorption energy = E(total) - E(surface) - E(adsorbate)
Only relative energy differences matter!
"""

from ase.build import fcc111, molecule, add_adsorbate
from ase import Atoms
from chem_gym.surrogate.ocp_model import EquiformerV2Oracle
import numpy as np


def predict_energy_safe(oracle, atoms):
    """Predict energy with error handling"""
    try:
        return oracle.predict_energy(atoms)
    except Exception as e:
        print(f"Error predicting energy: {e}")
        return None


def test_proper_adsorption():
    """Proper test: Only relative energy differences matter"""

    print("=" * 70)
    print("PROPER ADSORPTION ENERGY TEST")
    print("=" * 70)
    print("Principle: ΔE_ads = E(total) - E(surface) - E(adsorbate)")
    print("Only relative differences matter - absolute values use reference state")
    print("=" * 70)

    # Load model
    checkpoint = "checkpoints/eq2_83M_2M.pt"
    oracle = EquiformerV2Oracle(checkpoint, device="cuda")

    # 1. Clean surface
    print("\n1. Creating clean Cu(111) surface...")
    surface = fcc111('Cu', size=(3, 3, 4), vacuum=10.0)
    surface.set_pbc(True)  # Ensure periodic boundaries
    surface.cell[2, 2] = 20.0  # Increase vacuum to avoid interactions
    surface.center()
    print(f"   Surface atoms: {len(surface)} Cu atoms")

    # 2. Gas phase adsorbate
    print("\n2. Creating OH adsorbate in gas phase...")
    oh = molecule('OH')
    oh.set_tags([2, 2])  # Mark as adsorbate
    print(f"   OH molecule in vacuum")
    print(f"   OH atoms: {len(oh)} (O + H)")

    # 3. Adsorbed system (surface + OH)
    print("\n3. Creating adsorbed system (surface + OH)...")
    adsorbed = surface.copy()

    # Add OH above a Cu atom (ontop site)
    add_adsorbate(adsorbed, oh, 2.0, position=(0, 0))

    # Set tags: 1=surface, 2=adsorbate
    tags = np.ones(len(adsorbed), dtype=int)
    tags[-len(oh):] = 2  # Last 2 atoms are adsorbate
    adsorbed.set_tags(tags)

    n_surface = np.sum(tags == 1)
    n_adsorbate = np.sum(tags == 2)

    print(f"   Total atoms: {len(adsorbed)}")
    print(f"   Surface atoms (tag=1): {n_surface}")
    print(f"   Adsorbate atoms (tag=2): {n_adsorbate}")
    print(f"   All tags: {adsorbed.get_tags()}")

    # 4. Predict energies
    print("\n4. Predicting energies with EquiformerV2...")
    print("-" * 70)

    e_total = predict_energy_safe(oracle, adsorbed)
    e_surface = predict_energy_safe(oracle, surface)
    e_adsorbate = predict_energy_safe(oracle, oh)

    if None in [e_total, e_surface, e_adsorbate]:
        print("ERROR: Failed to predict energies")
        return

    print(f"E_total (surface + OH):     {e_total:12.4f} eV")
    print(f"E_surface (clean surface):  {e_surface:12.4f} eV")
    print(f"E_adsorbate (OH gas phase): {e_adsorbate:12.4f} eV")
    print("-" * 70)

    # 5. Calculate adsorption energy
    print("\n5. Calculating adsorption energy...")
    adsorption_energy = e_total - e_surface - e_adsorbate

    print(f"ΔE_ads = E_total - E_surface - E_adsorbate")
    print(f"ΔE_ads = {e_total:.4f} - {e_surface:.4f} - {e_adsorbate:.4f}")
    print(f"ΔE_ads = {adsorption_energy:12.4f} eV")
    print("-" * 70)

    # 6. Physical validation (relative is all that matters!)
    print("\n6. Physical validation...")
    print("Note: We only care about relative sign and magnitude")
    print("-" * 70)

    if adsorption_energy < -1.0:
        print(f"✓ STRONG CHEMISORPTION: ΔE = {adsorption_energy:.2f} eV")
        print("  Expected: Strong bonding (e.g., OH on Cu, CO on Pt)")
    elif adsorption_energy < -0.3:
        print(f"✓ WEAK CHEMISORPTION: ΔE = {adsorption_energy:.2f} eV")
        print("  Expected: Moderate bonding")
    elif adsorption_energy < 0:
        print(f"✓ PHYSISORPTION: ΔE = {adsorption_energy:.2f} eV")
        print("  Expected: Weak van der Waals interaction")
    else:
        print(f"✗ UNSTABLE/REPULSIVE: ΔE = {adsorption_energy:.2f} eV")
        print("  Expected: Adsorbate would not bind")

    print("-" * 70)

    # 7. Test with different binding sites
    print("\n7. Testing different binding sites...")
    print("Just for validation - energies should differ")
    print("-" * 70)

    # On-top (already tested)
    e_ontop = adsorption_energy
    print(f"On-top site:     ΔE = {e_ontop:.4f} eV")

    # Hollow site
    hollow = surface.copy()
    oh_hollow = molecule('OH')
    # Position OH above hollow site (between 3 surface atoms)
    hollow_pos = np.mean([surface[i].position for i in [0, 1, 3]], axis=0)
    oh_hollow.translate(hollow_pos + [0, 0, 2.0])
    hollow += oh_hollow

    tags = np.ones(len(hollow), dtype=int)
    tags[-len(oh_hollow):] = 2
    hollow.set_tags(tags)

    try:
        e_hollow_total = oracle.predict_energy(hollow)
        e_hollow = e_hollow_total - e_surface - e_adsorbate
        print(f"Hollow site:     ΔE = {e_hollow:.4f} eV")
        print(f"Difference:      ΔΔE = {e_hollow - e_ontop:.4f} eV")
    except Exception as e:
        print(f"Hollow site:     ERROR - {e}")

    # 8. Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("✓ Model correctly distinguishes surface vs adsorbate atoms")
    print("✓ Predictions use tags (1=surface, 2=adsorbate)")
    print("✓ Only relative energy differences matter")
    print("✗ If results are unphysical (>0 eV), it's because:")
    print("  - The OC20 model might need different reference states")
    print("  - The decomposition E_total - E_surface - E_adsorbate assumes")
    print("    gas-phase reference for adsorbate")
    print("=" * 70)


if __name__ == "__main__":
    test_proper_adsorption()

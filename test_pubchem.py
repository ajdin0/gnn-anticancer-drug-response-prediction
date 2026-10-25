import pubchempy as pcp

test_drugs = ["Erlotinib", "Cisplatin", "Paclitaxel", "Gefitinib", "5-Fluorouracil"]

for drug in test_drugs:
    print("\nDrug:", drug)
    try:
        compounds = pcp.get_compounds(drug, "name")
        print("Found:", len(compounds))

        if compounds:
            print("SMILES:", compounds[0].canonical_smiles)
    except Exception as e:
        print("Error:", e)
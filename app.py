# phytoai_complete_fixed.py
import streamlit as st
import pandas as pd
import numpy as np
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import Draw
import plotly.express as px
from st_aggrid import AgGrid, GridOptionsBuilder
from io import BytesIO
import time

st.set_page_config(
    page_title="PhytoAI Pro",
    layout="wide",
    page_icon=""
)
st.title("PhytoAI - Domain-Specific Analyzer")

DOMAIN_TARGETS = {
    "Oncology": ["TP53", "BRCA1", "EGFR", "HER2", "VEGF", "PDGFR"],
    "Neurology": ["APP", "MAPT", "SNCA", "PARK7", "HTT", "PSEN1"],
    "Immunology": ["IL6", "TNF", "IFNG", "IL1B", "IL17", "IL4"],
    "Infectious Disease": ["ACE2", "SPIKE", "RdRp", "PROTEASE", "NS5", "ENV"]
}

@st.cache_data(ttl=3600, show_spinner="Fetching domain-specific compounds...")
def fetch_domain_compounds(gene: str, domain: str, ic50_threshold: float, max_results: int = 30):
    try:
        gene = gene.upper().strip()
        if domain not in DOMAIN_TARGETS:
            raise ValueError(f"Invalid domain: {domain}")
        if gene not in DOMAIN_TARGETS[domain]:
            st.warning(f"{gene} is not a primary target for {domain}. Showing available results anyway.")

        targets = list(new_client.target.filter(
            target_synonym__iexact=gene
        ).only(['target_chembl_id', 'pref_name']))

        if not targets:
            return pd.DataFrame()

        activities = list(new_client.activity.filter(
            target_chembl_id=targets[0]['target_chembl_id'],
            standard_type="IC50",
            standard_units="nM",
            standard_value__lte=float(ic50_threshold)
        )[:max_results])

        results = []
        seen_molecules = set()

        for act in activities:
            try:
                mol_id = act['molecule_chembl_id']
                if mol_id in seen_molecules:
                    continue
                seen_molecules.add(mol_id)

                mol = new_client.molecule.get(mol_id)
                if not mol or not mol.get('molecule_structures'):
                    continue

                smiles = mol['molecule_structures'].get('canonical_smiles')
                if not smiles:
                    continue

                pathway = act.get('assay_description', 'Unknown')
                if domain == "Oncology":
                    pathway = pathway.replace("cancer", "anti-cancer").replace("tumor", "anti-tumor")

                name = mol.get('pref_name')
                if not name or name.lower().startswith("chembl"):
                    name = f"Compound {mol_id}"

                results.append({
                    "Name": name,
                    "SMILES": smiles,
                    "Target": gene,
                    "IC50_nM": float(act['standard_value']),
                    "Pathway": pathway,
                    "Domain": domain,
                    "ChEMBL_ID": mol_id,
                    "Type": mol.get("molecule_type", "unknown")
                })
            except Exception as e:
                continue

        return pd.DataFrame(results)

    except Exception as e:
        st.error(f"Data fetch failed: {str(e)}")
        return pd.DataFrame()

def render_molecule(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            img = Draw.MolToImage(mol, size=(350, 250))
            st.image(img, caption="2D Chemical Structure", use_column_width=True)
            st.caption(f"SMILES: `{smiles[:50]}{'...' if len(smiles) > 50 else ''}`")
        else:
            st.warning("Could not parse SMILES string")
    except Exception as e:
        st.error(f"Rendering failed: {str(e)}")

def render_heatmap(data: pd.DataFrame):
    try:
        if len(data) < 2:
            st.info("Need at least 2 compounds for heatmap visualization")
            return

        pivot_data = data.pivot_table(
            values='IC50_nM',
            index='Name',
            columns='Target',
            aggfunc=np.mean
        )

        fig = px.imshow(
            pivot_data,
            color_continuous_scale='Viridis',
            labels={'color': 'IC50 (nM)'},
            zmin=data['IC50_nM'].min() * 0.9,
            zmax=data['IC50_nM'].max() * 1.1
        )
        fig.update_layout(
            title=f"{data.iloc[0]['Domain']} Potency Comparison",
            xaxis_title="Target Gene",
            yaxis_title="Phytochemical"
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Heatmap failed: {str(e)}")

def create_sidebar():
    with st.sidebar:
        st.header("Query Parameters")

        domain = st.selectbox("Biomedical Domain", list(DOMAIN_TARGETS.keys()), index=0, key='domain_select')
        example_genes = ", ".join(DOMAIN_TARGETS[domain][:3])
        gene = st.text_input(f"Target Gene (e.g., {example_genes})", value=DOMAIN_TARGETS[domain][0], key='gene_input').upper().strip()
        ic50 = st.slider("Maximum IC50 (nM)", min_value=0.1, max_value=100000.0, value=10000.0, step=100.0, format="%.2f", key='ic50_slider')
        max_results = st.slider("Maximum Results", min_value=5, max_value=100, value=25, step=5, key='max_results')

        return domain, gene, ic50, max_results

def main():
    if 'selected_compound' not in st.session_state:
        st.session_state.selected_compound = None

    domain, gene, ic50, max_results = create_sidebar()

    if 'data' not in st.session_state:
        st.session_state.data = pd.DataFrame()

    auto_trigger = False
    if st.session_state.data.empty:
        auto_trigger = True

    if st.sidebar.button("Launch Analysis", type="primary", use_container_width=True) or auto_trigger:
        with st.spinner(f"Querying {domain} compounds for {gene}... Please wait."):
            start_time = time.time()
            data = fetch_domain_compounds(gene, domain, ic50, max_results)
            elapsed = time.time() - start_time

            if data.empty:
                st.error(f"No {domain.lower()} compounds found for {gene} at IC50 ≤ {ic50} nM")
                st.session_state.selected_compound = None
            else:
                st.success(f"Found {len(data)} {domain.lower()} compounds in {elapsed:.1f}s")
                st.session_state.data = data
                st.session_state.selected_compound = data.iloc[0].to_dict()

    if not st.session_state.data.empty:
        data = st.session_state.data

        tab1, tab2, tab3, tab4 = st.tabs(["Compounds Table", "Molecular Viewer", "Bio Details", "Potency Analysis"])

        with tab1:
            st.subheader(f"{domain} Compounds Targeting {gene}")

            gb = GridOptionsBuilder.from_dataframe(data)
            gb.configure_default_column(wrapText=True, autoHeight=True, resizable=True)

            gb.configure_column("Name", header_name="Phytochemical", width=250, tooltipField="Name")
            gb.configure_column("IC50_nM", header_name="IC50 (nM)", width=120, type=["numericColumn"])
            gb.configure_column("Pathway", header_name="Pathway", width=300, tooltipField="Pathway")
            gb.configure_column("ChEMBL_ID", header_name="ChEMBL ID", width=150)
            gb.configure_column("Type", header_name="Target Class", width=150)
            gb.configure_column("SMILES", hide=True)
            gb.configure_column("Target", hide=True)
            gb.configure_column("Domain", hide=True)

            gb.configure_selection("single", use_checkbox=True)

            grid_height = min(650, 60 + len(data) * 39)
            grid = AgGrid(
                data,
                gridOptions=gb.build(),
                height=grid_height,
                width='100%',
                theme="streamlit",
                update_mode='MODEL_CHANGED',
                allow_unsafe_jscode=True
            )

            if grid['selected_rows']:
                st.session_state.selected_compound = grid['selected_rows'][0]

        with tab2:
            st.subheader("Molecular Structure")
            if st.session_state.selected_compound:
                render_molecule(st.session_state.selected_compound['SMILES'])
            else:
                st.info("Select a compound in the table to view structure")

        with tab3:
            st.subheader("Biological Details")
            if st.session_state.selected_compound:
                compound = st.session_state.selected_compound
                st.markdown(f"""
                ### {compound['Name']}
                **Target**: `{compound['Target']}`  
                **Domain**: `{compound['Domain']}`  
                **Potency**: `{compound['IC50_nM']:.2f} nM`  
                **Primary Pathway**: {compound['Pathway']}  
                **Molecule Type**: `{compound['Type']}`  

                [View on ChEMBL](https://www.ebi.ac.uk/chembl/compound_report_card/{compound['ChEMBL_ID']}/)
                """)
            else:
                st.info("Select a compound for detailed view")

        with tab4:
            st.subheader("Domain-Specific Potency")
            render_heatmap(data)

            csv = data.to_csv(index=False).encode()
            st.download_button(
                "Download All Compounds",
                data=csv,
                file_name=f"phytoai_{domain}_{gene}_compounds.csv",
                mime="text/csv",
                use_container_width=True
            )

if __name__ == "__main__":
    main()


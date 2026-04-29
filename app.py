import streamlit as st
from sentinelhub import SHConfig, SentinelHubRequest, DataCollection, BBox, CRS, MimeType
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import datetime
from matplotlib.path import Path

CDSE_S2L2A = DataCollection.define(
    name="Sentinel-2 L2A CDSE",
    api_id="sentinel-2-l2a",
    service_url="https://sh.dataspace.copernicus.eu"
)

# --- 1. KONFIGURACJA KLUCZY ---
config = SHConfig()
config.sh_client_id = st.secrets["SH_CLIENT_ID"]
config.sh_client_secret = st.secrets["SH_CLIENT_SECRET"]
config.sh_base_url = 'https://sh.dataspace.copernicus.eu'
config.sh_token_url = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
config.save()

# --- 2. GEOMETRIA POLA (40 ha) ---
coords = [[21.50438, 50.835016], [21.506048, 50.834379], [21.511863, 50.839586], [21.504595, 50.84301], [21.506349, 50.844484], [21.504836, 50.844436], [21.500888, 50.845243], [21.499772, 50.844497], [21.500373, 50.843292], [21.503549, 50.840893], [21.50754, 50.839037], [21.506939, 50.838536], [21.505458, 50.839335], [21.503463, 50.839769], [21.500909, 50.841557], [21.500416, 50.841435], [21.501102, 50.83863], [21.502068, 50.838102], [21.501746, 50.837167], [21.502326, 50.836693], [21.503463, 50.836232], [21.504493, 50.837275], [21.506145, 50.836679], [21.50438, 50.835016]]

min_x = min([c[0] for c in coords])
max_x = max([c[0] for c in coords])
min_y = min([c[1] for c in coords])
max_y = max([c[1] for c in coords])
bbox = BBox(bbox=[min_x, min_y, max_x, max_y], crs=CRS.WGS84)

# --- 3. INTERFEJS UŻYTKOWNIKA ---
st.set_page_config(page_title="Monitor Pola Taty", layout="centered")
st.title("🚜 Monitoring Pola: 40 ha")

st.sidebar.header("Ustawienia")
data_widoku = st.sidebar.date_input("Wybierz datę analizy", value=datetime.date(2026, 4, 15))
st.sidebar.markdown("---")
# DODANA NOWA OPCJA DO MENU!
typ_analizy = st.sidebar.radio(
    "Wybierz rodzaj analizy:",
    ("Kondycja roślin (NDVI)", "Wilgotność (NDWI)", "Strefy nawożenia (Zoning)")
)

if st.button('Pobierz analizę z satelity'):
    with st.spinner('Łączę się z satelitą Sentinel-2...'):
        try:
            start_dt = (data_widoku - datetime.timedelta(days=15)).strftime('%Y-%m-%d')
            end_dt = data_widoku.strftime('%Y-%m-%d')

            evalscript = """
            //VERSION=3
            function setup() {
              return {
                input: ["B02", "B03", "B04", "B08", "B11"],
                output: { bands: 5, sampleType: "FLOAT32" }
              };
            }
            function evaluatePixel(sample) {
              let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
              let ndwi = (sample.B08 - sample.B11) / (sample.B08 + sample.B11);
              return [sample.B04 * 2.5, sample.B03 * 2.5, sample.B02 * 2.5, ndvi, ndwi];
            }
            """

            request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=CDSE_S2L2A,
                        time_interval=(start_dt, end_dt),
                    )
                ],
                responses=[
                    SentinelHubRequest.output_response('default', MimeType.TIFF)
                ],
                bbox=bbox,
                size=[600, 600],
                config=config
            )

            fetched_data = request.get_data()
            
            if not fetched_data:
                st.warning("Brak bezchmurnych zdjęć w tym okresie. Spróbuj wybrać późniejszą datę.")
            else:
                raw_data = fetched_data[0]
                
                rgb_image = np.clip(raw_data[:, :, :3], 0, 1)
                grayscale_back = np.dot(rgb_image[...,:3], [0.299, 0.587, 0.114])
                grayscale_rgb = np.stack((grayscale_back,)*3, axis=-1)
                
                # MASKOWANIE (Robimy to na początku, żeby algorytm nawozów liczył tylko pole, a nie drogę)
                pixel_coords = []
                for lon, lat in coords:
                    px = (lon - min_x) / (max_x - min_x) * 600
                    py = 600 - ((lat - min_y) / (max_y - min_y) * 600) 
                    pixel_coords.append([px, py])
                    
                x, y = np.meshgrid(np.arange(600), np.arange(600))
                points = np.vstack((x.flatten(), y.flatten())).T
                path = Path(pixel_coords)
                mask = path.contains_points(points).reshape(600, 600)

                uzyj_stref = False # Flaga sprawdzająca, czy używamy twardych stref

                # --- WYBÓR NAKŁADKI ---
                if typ_analizy == "Kondycja roślin (NDVI)":
                    mapa_warstwa = raw_data[:, :, 3].copy()
                    mapa_warstwa[~mask] = np.nan
                    paleta_kolorow = 'RdYlGn'
                    zakres_min, zakres_max = 0.0, 1.0
                    interpolacja = 'bicubic'
                    tytul_mapy = f"Zdrowie Roślin (NDVI) | {start_dt} do {end_dt}"
                    info_tekst = "💡 Intensywny zielony to mocna wegetacja. Czerwony to problem."

                elif typ_analizy == "Wilgotność (NDWI)":
                    mapa_warstwa = raw_data[:, :, 4].copy()
                    mapa_warstwa[~mask] = np.nan
                    paleta_kolorow = 'BrBG'
                    zakres_min, zakres_max = -0.2, 0.6
                    interpolacja = 'bicubic'
                    tytul_mapy = f"Wilgotność Terenu (NDWI) | {start_dt} do {end_dt}"
                    info_tekst = "💧 Ciemny turkus to mokro. Brąz to sucho."

                else: # STREFY NAWOŻENIA (Nowa magia!)
                    ndvi_raw = raw_data[:, :, 3].copy()
                    ndvi_raw[~mask] = np.nan # Liczymy statystyki TYLKO dla pola
                    
                    # Matematyka: Dzielimy pole na 3 idealne strefy (Percentyle 33% i 66%)
                    p33 = np.nanpercentile(ndvi_raw, 33)
                    p66 = np.nanpercentile(ndvi_raw, 66)
                    
                    strefy = np.zeros_like(ndvi_raw)
                    strefy[ndvi_raw <= p33] = 1 # Słaba
                    strefy[(ndvi_raw > p33) & (ndvi_raw <= p66)] = 2 # Średnia
                    strefy[ndvi_raw > p66] = 3 # Mocna
                    strefy[~mask] = np.nan # Wyciszamy drogę
                    
                    mapa_warstwa = strefy
                    # Tworzymy twardą paletę kolorów dla nawozów
                    paleta_kolorow = mcolors.ListedColormap(['#d73027', '#fee08b', '#1a9850']) # Czerw, Żółty, Zielony
                    bounds = [0.5, 1.5, 2.5, 3.5]
                    norm = mcolors.BoundaryNorm(bounds, paleta_kolorow.N)
                    uzyj_stref = True
                    interpolacja = 'nearest' # Celowo wyłączamy wygładzanie! Rozsiewacz to nie pędzel.
                    tytul_mapy = f"Strefy Nawożenia (VRA) | {start_dt} do {end_dt}"
                    info_tekst = "🚜 Strefa Czerwona (1) to najsłabszy wigor. Strefa Zielona (3) to rośliny w super kondycji."
                
                poly_path = np.array(pixel_coords)
                
                # --- RYSOWANIE MAPY FINALNEJ ---
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(grayscale_rgb, interpolation='bicubic')
                
                if uzyj_stref:
                    # Specjalne rysowanie dla twardych stref z odpowiednią legendą
                    im = ax.imshow(mapa_warstwa, cmap=paleta_kolorow, norm=norm, alpha=0.85, interpolation=interpolacja)
                    cbar = plt.colorbar(im, fraction=0.046, pad=0.04, ticks=[1, 2, 3])
                    cbar.set_ticklabels(['Strefa 1\n(Słaby wigor)', 'Strefa 2\n(Średniak)', 'Strefa 3\n(Mocny wigor)'])
                    cbar.set_label('Rekomendacja działania')
                else:
                    # Rysowanie wygładzone dla NDVI i NDWI
                    im = ax.imshow(mapa_warstwa, cmap=paleta_kolorow, vmin=zakres_min, vmax=zakres_max, alpha=0.9, interpolation=interpolacja)
                    etykieta = 'Indeks NDVI (Zielony=Zdrowe)' if typ_analizy == "Kondycja roślin (NDVI)" else 'Indeks NDWI (Niebieski=Mokro)'
                    plt.colorbar(im, label=etykieta, fraction=0.046, pad=0.04) 
                
                ax.plot(poly_path[:, 0], poly_path[:, 1], color='black', linewidth=1.5, alpha=0.8)
                ax.set_title(tytul_mapy, fontsize=14, fontweight='bold')
                ax.axis('off')
                fig.patch.set_facecolor('white')
                
                st.pyplot(fig)
                st.success("Analiza wykonana pomyślnie!")
                st.info(info_tekst)

        except Exception as e:
            st.error(f"Błąd podczas pobierania danych: {e}")
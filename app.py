import streamlit as st
from sentinelhub import SHConfig, SentinelHubRequest, DataCollection, BBox, CRS, MimeType
import numpy as np
import matplotlib.pyplot as plt
import datetime
from matplotlib.path import Path

CDSE_S2L2A = DataCollection.define(
    name="Sentinel-2 L2A CDSE",
    api_id="sentinel-2-l2a",
    service_url="https://sh.dataspace.copernicus.eu"
)
# --- 1. KONFIGURACJA KLUCZY ---
# Pamiętaj: Client Secret widzisz tylko raz przy tworzeniu klucza!

config = SHConfig()
config.sh_client_id = 'sh-f48096a8-42c9-40af-8c8f-db4eb69b1422'
config.sh_client_secret = 'B3VX7e8IRf8yaSsmo9Z6GIJhSKDEwdCo'
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
# Domyślnie ustawiamy datę na dzisiejszą (kwiecień 2026)
data_widoku = st.sidebar.date_input("Wybierz datę analizy", value=datetime.date(2026, 4, 15))

if st.button('Pobierz stan roślin (NDVI)'):
    with st.spinner('Łączę się z satelitą Sentinel-2...'):
        try:
            # Określamy zakres czasu (+/- 15 dni od wybranej daty)
            start_dt = (data_widoku - datetime.timedelta(days=15)).strftime('%Y-%m-%d')
            end_dt = data_widoku.strftime('%Y-%m-%d')

            evalscript = """
            //VERSION=3
            function setup() {
              return {
                input: ["B02", "B03", "B04", "B08"],
                output: { bands: 4, sampleType: "FLOAT32" }
              };
            }
            function evaluatePixel(sample) {
              let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
              // Zwracamy pasma RGB (rozjaśnione *2.5) oraz NDVI w czwartej warstwie
              return [sample.B04 * 2.5, sample.B03 * 2.5, sample.B02 * 2.5, ndvi];
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
                
                # --- PRZYGOTOWANIE TŁA CZARNO-BIAŁEGO ---
                # Wyciągamy kolory widzialne i wycinamy do zakresu 0-1
                rgb_image = np.clip(raw_data[:, :, :3], 0, 1)
                
                # Standardowa formuła matematyczna na zamianę RGB na odcienie szarości
                # (Luminancja: 0.299 * Czerwony + 0.587 * Zielony + 0.114 * Niebieski)
                grayscale_back = np.dot(rgb_image[...,:3], [0.299, 0.587, 0.114])
                
                # Zamieniamy szarą mapę znowu na format RGB (duplikujemy kanał szarości 3 razy)
                # żeby matplotlib mógł go narysować jako zdjęcie.
                grayscale_rgb = np.stack((grayscale_back,)*3, axis=-1)
                
                # --- PRZYGOTOWANIE NAKŁADKI NDVI ---
                # Wyciągamy samą nakładkę NDVI (4 warstwa)
                ndvi_image = raw_data[:, :, 3].copy()
                
                # --- MASKOWANIE POLA (To co mieliśmy wcześniej) ---
                pixel_coords = []
                for lon, lat in coords:
                    px = (lon - min_x) / (max_x - min_x) * 600
                    py = 600 - ((lat - min_y) / (max_y - min_y) * 600) 
                    pixel_coords.append([px, py])
                    
                x, y = np.meshgrid(np.arange(600), np.arange(600))
                points = np.vstack((x.flatten(), y.flatten())).T
                
                path = Path(pixel_coords)
                mask = path.contains_points(points).reshape(600, 600)
                
                # Wszędzie tam, gdzie NIE MA pola, ustawiamy nakładkę NDVI na przezroczystą (NaN)
                ndvi_image[~mask] = np.nan
                
                # --- PROFI OBKREŚLENIE GRANIC POLA (Linia od linijki, ale celowa) ---
                # Dodajemy cienką, czarną linię graniczną na NDVI, żeby zamaskować piksele
                poly_path = np.array(pixel_coords)
                
                # -------------------------------
                
                # --- RYSOWANIE MAPY FINALNEJ ---
                fig, ax = plt.subplots(figsize=(10, 10)) # Trochę większe zdjęcie
                
                # 1. Kładziemy tło CZARNO-BIAŁE (cała okolica)
                ax.imshow(grayscale_rgb, interpolation='bicubic')
                
                # 2. Kładziemy KOLOROWĄ nakładkę NDVI (tylko na pole)
                # Ustawiamy 'alpha=0.9', żeby kolory były nasycone, ale lekko przebijała struktura ziemi
                im = ax.imshow(ndvi_image, cmap='RdYlGn', vmin=0, vmax=1, alpha=0.9, interpolation='bicubic')
                
                # 3. Dodajemy PROFI CZARNĄ LINIĘ na granice, żeby maska wyglądała ostro
                ax.plot(poly_path[:, 0], poly_path[:, 1], color='black', linewidth=1.5, alpha=0.8)
                
                # Pasek z kolorami (fraction=0.046 pad=0.04 to złote proporcje, żeby nie psuł układu)
                plt.colorbar(im, label='Indeks NDVI (Zielony = Zdrowe)', fraction=0.046, pad=0.04) 
                
                ax.set_title(f"Monitor Pola (NDVI) | Okres: {start_dt} do {end_dt}", fontsize=14, fontweight='bold')
                ax.axis('off')
                
                # Poprawka estetyczna, żeby białe brzegi wykresu zniknęły
                fig.patch.set_facecolor('white')
                
                st.pyplot(fig)
                st.success("Nowa, stylistyczna mapa wygenerowana pomyślnie!")
                st.info("💡 Interpretacja: Kolorowa nakładka pokazuje zdrowie uprawy na Twoim polu. Otoczenie jest czarno-białe dla kontekstu.")

        except Exception as e:
            st.error(f"Błąd podczas pobierania danych: {e}")
            if "invalid_client" in str(e):
                st.error("⚠️ PROBLEM Z KLUCZAMI: Wygeneruj nowy Client ID i Client Secret w panelu CDSE!")
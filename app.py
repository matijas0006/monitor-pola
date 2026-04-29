import streamlit as st
from sentinelhub import SHConfig, SentinelHubRequest, DataCollection, BBox, CRS, MimeType
import numpy as np
import matplotlib.pyplot as plt
import datetime

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
                input: ["B04", "B08"],
                output: { bands: 1, sampleType: "FLOAT32" }
              };
            }
            function evaluatePixel(sample) {
              let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
              return [ndvi];
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
                data = fetched_data[0]
                
                # Rysowanie mapy
                fig, ax = plt.subplots(figsize=(8, 8))
                im = ax.imshow(data, cmap='RdYlGn', vmin=0, vmax=1)
                plt.colorbar(im, label='Indeks NDVI (Zielony = Zdrowe)')
                ax.set_title(f"Kondycja pola w okresie {start_dt} do {end_dt}")
                ax.axis('off')
                
                st.pyplot(fig)
                st.success("Mapa wygenerowana pomyślnie!")
                st.info("💡 Interpretacja: Intensywny zielony to mocne wschody. Czerwony to goła ziemia lub problem.")

        except Exception as e:
            st.error(f"Błąd podczas pobierania danych: {e}")
            if "invalid_client" in str(e):
                st.error("⚠️ PROBLEM Z KLUCZAMI: Wygeneruj nowy Client ID i Client Secret w panelu CDSE!")
import streamlit as st
from sentinelhub import SHConfig, SentinelHubRequest, DataCollection, BBox, CRS, MimeType, SentinelHubCatalog
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import datetime
from matplotlib.path import Path

# --- 1. DEFINICJA BAZY DANYCH ---
CDSE_S2L2A = DataCollection.define(
    name="Sentinel-2 L2A CDSE",
    api_id="sentinel-2-l2a",
    service_url="https://sh.dataspace.copernicus.eu"
)

# --- 2. KONFIGURACJA KLUCZY ---
config = SHConfig()
config.sh_client_id = st.secrets["SH_CLIENT_ID"]
config.sh_client_secret = st.secrets["SH_CLIENT_SECRET"]
config.sh_base_url = 'https://sh.dataspace.copernicus.eu'
config.sh_token_url = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
config.save()

# --- WSPÓŁRZĘDNE DLA ANTENY I KAROLKA ---
coords_anteny = [[21.49258923665093, 50.83754909442891], [21.493324435086322, 50.83793160969614], [21.49181957976859, 50.839093231344776], [21.492524383705728, 50.839495276653935], [21.491142253914035, 50.84053538278707], [21.48977843308859, 50.8414431140184], [21.489421822770055, 50.841247979768525], [21.489411949714196, 50.84104593823196], [21.489784488343417, 50.84074289376369], [21.490205749096305, 50.8404326472978], [21.490228399646014, 50.84030476836554], [21.48969162096799, 50.84060419122491], [21.48958871309128, 50.840622048136964], [21.489438512095518, 50.84059567305374], [21.489162758889023, 50.84049683361039], [21.488913305778684, 50.84038596682663], [21.488991532082395, 50.84024771055917], [21.489049268622068, 50.8401058867552], [21.48942334764888, 50.839798959719104], [21.49110153167493, 50.83855101601819], [21.492219242006342, 50.83775020410957], [21.492511464113363, 50.83751169170253], [21.49258923665093, 50.83754909442891]]
coords_karolka = [[21.497709472129515, 50.83741324313428], [21.497905331455996, 50.837700315212714], [21.498056286177302, 50.83785929293714], [21.498482578237628, 50.83864194636806], [21.49672640669894, 50.839226783896805], [21.496552410935976, 50.839048250877624], [21.495706172553724, 50.838624844861215], [21.4956992739167, 50.838551261003715], [21.496514083474807, 50.83806408156974], [21.49764493529051, 50.83742726783569], [21.497709472129515, 50.83741324313428]]

# --- 3. BAZA WASZYCH PÓL ---
POLA = {
    "Czterdziestka (40 ha)": {"polys": [
        [[21.50438, 50.835016], [21.506048, 50.834379], [21.511863, 50.839586], [21.504595, 50.84301], [21.506349, 50.844484], [21.504836, 50.844436], [21.500888, 50.845243], [21.499772, 50.844497], [21.500373, 50.843292], [21.503549, 50.840893], [21.50754, 50.839037], [21.506939, 50.838536], [21.505458, 50.839335], [21.503463, 50.839769], [21.500909, 50.841557], [21.500416, 50.841435], [21.501102, 50.83863], [21.502068, 50.838102], [21.501746, 50.837167], [21.502326, 50.836693], [21.503463, 50.836232], [21.504493, 50.837275], [21.506145, 50.836679], [21.50438, 50.835016]]
    ], "labels": []},
    "18": {"polys": [
        [[21.5115216811856, 50.839899971578006], [21.513899058883936, 50.842431983901434], [21.51222165910471, 50.84346024979146], [21.510943361311462, 50.84385794359682], [21.510024754837616, 50.844010822403334], [21.509077865113852, 50.84420857870376], [21.50671369961276, 50.84477062645118], [21.506065690394934, 50.844374945542825], [21.505253898625995, 50.84363753125834], [21.50492633352718, 50.84324633724185], [21.504954817448095, 50.84312942804834], [21.506979444661738, 50.84212175092563], [21.5115216811856, 50.839899971578006]]
    ], "labels": []},
    "Koło winnicy": {"polys": [
        [[21.50461724408359, 50.85132033676828], [21.503192876674774, 50.848666487865586], [21.50153105699104, 50.84527974830996], [21.502982785485926, 50.84501798507509], [21.506075382822473, 50.85116955286688], [21.50461724408359, 50.85132033676828]],
        [[21.510442549929763, 50.850765479013006], [21.50989219582513, 50.84983485862867], [21.508699393541093, 50.847341467261884], [21.509009833251582, 50.84728359144805], [21.509526243998494, 50.84793762144986], [21.51014102548632, 50.848442636462266], [21.510438554116604, 50.84868651190092], [21.51091728516454, 50.849239065613176], [21.511376649021514, 50.84996937908636], [21.511475768774886, 50.85059229990733], [21.510442549929763, 50.850765479013006]]
    ], "labels": []},
    "Koło anteny i karolka": {
        "polys": [coords_anteny, coords_karolka],
        "labels": [
            {"text": "ANTENA", "coords": [21.491, 50.8415]}, 
            {"text": "KAROLEK", "coords": [21.497, 50.8393]}
        ]
    },
    "Za Jurkiem": {"polys": [
        [[21.513201893908047, 50.83702919141547], [21.50899829160332, 50.83469087820191], [21.509517985780064, 50.83442570291723], [21.50925029294811, 50.83427884220015], [21.509633342433915, 50.83399139701305], [21.509235100491424, 50.83377818247041], [21.5097644819449, 50.83347617660007], [21.509224626661336, 50.833186470945094], [21.509537653376128, 50.83297085548409], [21.507873403646386, 50.83200654688616], [21.506293074630037, 50.83117332678884], [21.505128574220294, 50.830433093085645], [21.507333764133648, 50.828862017698924], [21.50931401619428, 50.827523176632326], [21.510300728367724, 50.827841149820756], [21.51179090581155, 50.82817379300644], [21.512910791383007, 50.828560447059516], [21.513747757330265, 50.829049325439314], [21.51403193400367, 50.82911980662618], [21.51518361634075, 50.829283082683844], [21.515725366577755, 50.82945017943956], [21.51622110277927, 50.82964514448787], [21.516806587965732, 50.829975783757845], [21.517402796548623, 50.83044058301189], [21.517563728732256, 50.8306683474259], [21.51784046177056, 50.83080680067371], [21.518087746760386, 50.83080680067371], [21.51916044904675, 50.83063265555933], [21.520953736746122, 50.83047242860914], [21.522331194686984, 50.83042686706261], [21.52126191983436, 50.83238300194688], [21.519245981324076, 50.83345117445751], [21.513201893908047, 50.83702919141547]],
        [[21.51368139062069, 50.836797705754265], [21.514382129771008, 50.83637238879675], [21.51745183923802, 50.83807236819234], [21.516744547324834, 50.83848355528005], [21.51368139062069, 50.836797705754265]],
        [[21.515170771066664, 50.83591519153683], [21.5158514784855, 50.83549757940983], [21.518895243588474, 50.83721608493474], [21.518205453549825, 50.83761226143935], [21.515170771066664, 50.83591519153683]],
        [[21.516992651931446, 50.83484714261755], [21.517720010494344, 50.834406511195795], [21.520686138190882, 50.83603383646732], [21.520614484966814, 50.8361843436555], [21.520096217154503, 50.83654496998034], [21.516992651931446, 50.83484714261755]],
        [[21.51815978405014, 50.8341642744349], [21.518586891979766, 50.83390258794722], [21.521043926064976, 50.835237323225925], [21.520881071230832, 50.835637035685494], [21.51815978405014, 50.8341642744349]],
        [[21.519134857741932, 50.833610421457536], [21.52097111937286, 50.83260758277075], [21.52134244366715, 50.833003206578184], [21.52148186277259, 50.833457948754074], [21.521535092328946, 50.833640122566834], [21.521517658373682, 50.833919303917725], [21.521269620381844, 50.8347708568667], [21.519792510058153, 50.8340315474658], [21.519134857741932, 50.833610421457536]]
    ], "labels": []}
}

# --- 4. INTERFEJS UŻYTKOWNIKA ---
st.set_page_config(page_title="Monitor Pola Taty", layout="centered")

opcje_menu = ["WSZYSTKIE POLA (Widok ogólny)"] + list(POLA.keys())

st.sidebar.header("Wybór Pola")
wybrane_nazwa = st.sidebar.selectbox("Wybierz pole do analizy:", opcje_menu)

# Wyznaczamy jakie poligony i etykiety mamy wyświetlić
aktywne_etykiety = []
if wybrane_nazwa == "WSZYSTKIE POLA (Widok ogólny)":
    wszystkie_poligony_list = [poly for p_data in POLA.values() for poly in p_data["polys"]]
    analiza_dostepna = False
else:
    wszystkie_poligony_list = POLA[wybrane_nazwa]["polys"]
    aktywne_etykiety = POLA[wybrane_nazwa]["labels"]
    analiza_dostepna = True

st.title(f"🚜 {wybrane_nazwa}")

st.sidebar.markdown("---")
st.sidebar.header("Ustawienia")
data_widoku = st.sidebar.date_input("Wybierz datę analizy", value=datetime.date.today())

if analiza_dostepna:
    st.sidebar.markdown("---")
    typ_analizy = st.sidebar.radio(
        "Wybierz rodzaj analizy:",
        ("Kondycja roślin (NDVI)", "Wilgotność (NDWI)", "Strefy nawożenia (Zoning)")
    )
else:
    st.sidebar.info("💡 W widoku ogólnym analiza jest wyłączona. Wybierz konkretne pole, aby sprawdzić NDVI/NDWI.")
    typ_analizy = "Naturalne kolory (RGB)"

# --- 5. OBLICZANIE RAMKI (BBOX) DLA WYBRANEGO OBSZARU ---
wszystkie_lon = [punkt[0] for poligon in wszystkie_poligony_list for punkt in poligon]
wszystkie_lat = [punkt[1] for poligon in wszystkie_poligony_list for punkt in poligon]

min_x, max_x = min(wszystkie_lon), max(wszystkie_lon)
min_y, max_y = min(wszystkie_lat), max(wszystkie_lat)

bufor_x = (max_x - min_x) * 0.1
bufor_y = (max_y - min_y) * 0.1
bbox = BBox(bbox=[min_x - bufor_x, min_y - bufor_y, max_x + bufor_x, max_y + bufor_y], crs=CRS.WGS84)

img_min_x, img_max_x = min_x - bufor_x, max_x + bufor_x
img_min_y, img_max_y = min_y - bufor_y, max_y + bufor_y

if st.button('Pobierz obraz z satelity'):
    with st.spinner('Przeszukuję archiwum satelity...'):
        try:
            start_dt = (data_widoku - datetime.timedelta(days=15)).strftime('%Y-%m-%d')
            end_dt = data_widoku.strftime('%Y-%m-%d')

            catalog = SentinelHubCatalog(config=config)
            search_iterator = catalog.search(collection=CDSE_S2L2A, bbox=bbox, time=(start_dt, end_dt))
            wszystkie_przeloty = list(search_iterator)
            
            dobre_zdjecia = [p for p in wszystkie_przeloty if p['properties'].get('eo:cloud_cover', 100) < 20]
            
            if not dobre_zdjecia:
                st.warning(f"Niestety, same grube chmury. Wybierz inną datę.")
                st.stop()
                
            dobre_zdjecia.sort(key=lambda x: x['properties']['datetime'], reverse=True)
            najlepszy_przelot = dobre_zdjecia[0]
            dokladna_data = najlepszy_przelot['properties']['datetime'][:10]
            st.success(f"Pobieram stan z dnia: **{dokladna_data}**")

            evalscript = """
            //VERSION=3
            function setup() {
              return { input: ["B02", "B03", "B04", "B08", "B11"], output: { bands: 5, sampleType: "FLOAT32" } };
            }
            function evaluatePixel(sample) {
              let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
              let ndwi = (sample.B08 - sample.B11) / (sample.B08 + sample.B11);
              return [sample.B04 * 2.5, sample.B03 * 2.5, sample.B02 * 2.5, ndvi, ndwi];
            }
            """

            request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[SentinelHubRequest.input_data(data_collection=CDSE_S2L2A, time_interval=(dokladna_data, dokladna_data))],
                responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
                bbox=bbox, size=[600, 600], config=config
            )

            fetched_data = request.get_data()
            
            if fetched_data:
                raw_data = fetched_data[0]
                rgb_image = np.clip(raw_data[:, :, :3], 0, 1)
                
                # --- PRZYGOTOWANIE MASKI ---
                maska_globalna = np.zeros((600, 600), dtype=bool)
                x_grid, y_grid = np.meshgrid(np.arange(600), np.arange(600))
                points = np.vstack((x_grid.flatten(), y_grid.flatten())).T
                sciezki_do_rysowania = []
                
                for poligon in wszystkie_poligony_list:
                    pixel_coords = []
                    for lon, lat in poligon:
                        px = (lon - img_min_x) / (img_max_x - img_min_x) * 600
                        py = 600 - ((lat - img_min_y) / (img_max_y - img_min_y) * 600) 
                        pixel_coords.append([px, py])
                        
                    path = Path(pixel_coords)
                    poly_mask = path.contains_points(points).reshape(600, 600)
                    maska_globalna = maska_globalna | poly_mask
                    sciezki_do_rysowania.append(np.array(pixel_coords))

                # --- RYSOWANIE GRAFIKI ---
                fig, ax = plt.subplots(figsize=(10, 10))
                
                if not analiza_dostepna:
                    ax.imshow(rgb_image, interpolation='bicubic')
                    info_tekst = "🗺️ Mapa poglądowa wszystkich pól w naturalnych kolorach. Granice zaznaczono na żółto."
                else:
                    grayscale_back = np.dot(rgb_image[...,:3], [0.299, 0.587, 0.114])
                    grayscale_rgb = np.stack((grayscale_back,)*3, axis=-1)
                    ax.imshow(grayscale_rgb, interpolation='bicubic')
                    
                    if typ_analizy == "Kondycja roślin (NDVI)":
                        mapa_warstwa = raw_data[:, :, 3].copy()
                        mapa_warstwa[~maska_globalna] = np.nan
                        im = ax.imshow(mapa_warstwa, cmap='RdYlGn', vmin=0.0, vmax=1.0, alpha=0.9, interpolation='bicubic')
                        plt.colorbar(im, label='Indeks NDVI (Zdrowie)', fraction=0.046, pad=0.04)
                        info_tekst = f"💡 Intensywny zielony to mocna wegetacja. Czerwony to problem lub goła ziemia."

                    elif typ_analizy == "Wilgotność (NDWI)":
                        mapa_warstwa = raw_data[:, :, 4].copy()
                        mapa_warstwa[~maska_globalna] = np.nan
                        im = ax.imshow(mapa_warstwa, cmap='BrBG', vmin=-0.2, vmax=0.6, alpha=0.9, interpolation='bicubic')
                        plt.colorbar(im, label='Indeks NDWI (Woda w roślinach)', fraction=0.046, pad=0.04)
                        info_tekst = f"💧 Ciemny turkus to woda w komórkach/błoto. Brąz to postępująca susza."

                    else: # ZONING
                        ndvi_raw = raw_data[:, :, 3].copy()
                        ndvi_raw[~maska_globalna] = np.nan
                        p33 = np.nanpercentile(ndvi_raw, 33)
                        p66 = np.nanpercentile(ndvi_raw, 66)
                        strefy = np.zeros_like(ndvi_raw)
                        strefy[ndvi_raw <= p33] = 1 
                        strefy[(ndvi_raw > p33) & (ndvi_raw <= p66)] = 2 
                        strefy[ndvi_raw > p66] = 3 
                        strefy[~maska_globalna] = np.nan
                        cmap_strefy = mcolors.ListedColormap(['#d73027', '#fee08b', '#1a9850'])
                        im = ax.imshow(strefy, cmap=cmap_strefy, alpha=0.85, interpolation='nearest')
                        cbar = plt.colorbar(im, fraction=0.046, pad=0.04, ticks=[1, 2, 3])
                        cbar.set_ticklabels(['Strefa 1\n(Słaby wigor)', 'Strefa 2\n(Średniak)', 'Strefa 3\n(Mocny wigor)'])
                        info_tekst = f"🚜 Strefy VRA dla rozsiewacza. UWAGA: Stosuj tylko, gdy na NDVI widać wyraźne różnice!"

                for poly_path in sciezki_do_rysowania:
                    kolor_linii = 'yellow' if not analiza_dostepna else 'black'
                    ax.plot(poly_path[:, 0], poly_path[:, 1], color=kolor_linii, linewidth=1.5, alpha=0.8)
                    
                # Rysowanie ewentualnych etykiet (podpisów) dla połączonych działek
                for etykieta in aktywne_etykiety:
                    px_text = (etykieta["coords"][0] - img_min_x) / (img_max_x - img_min_x) * 600
                    py_text = 600 - ((etykieta["coords"][1] - img_min_y) / (img_max_y - img_min_y) * 600)
                    ax.text(px_text, py_text, etykieta["text"], color='white', fontsize=11, fontweight='bold', ha='center', va='center', bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2'))
                    
                ax.axis('off')
                fig.patch.set_facecolor('white')
                
                st.pyplot(fig)
                st.info(info_tekst)

        except Exception as e:
            st.error(f"Wystąpił błąd podczas analizy. Błąd: {e}")
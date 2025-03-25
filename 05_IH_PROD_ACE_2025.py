# ====================
# 1. IMPORTACIONES
# ====================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import requests
from io import BytesIO

# ====================
# 2. CONFIGURACIÓN
# ====================
st.set_page_config("Índice de Heterogeneidad", layout="wide")
st.title("ÍNDICE DE HETEROGENEIDAD ACE")

# Insertar estilos con Markdown + HTML
st.markdown(
    """
    <style>
    body {
        background-color: #DDDDDD; /* Color de fondo claro */
    }
    .stApp {
        background-color: #DDDDDD; /* Fondo de toda la app */
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ====================
# 3. FUNCIONES AUXILIARES
# ====================

@st.cache_data
def cargar_datos(file):
    df = pd.read_csv(file)
    df['FECHA'] = pd.to_datetime(df['FECHA'], errors="coerce", dayfirst=True)
    return df

def validar_columnas(df, columnas):
    return [col for col in columnas if col not in df.columns]

def filtrar_pozos_con_fecha(df):
    pozos_validos = df.dropna(subset=["FECHA"])["POZO OFICIAL"].unique()
    return df[df["POZO OFICIAL"].isin(pozos_validos)].copy()

def secuencia_completa_fechas(df):  
    if df.empty or df["FECHA"].isna().all():
        return pd.DataFrame()
    
    fecha_min = df["FECHA"].min()
    fecha_max = df["FECHA"].max()
    fecha_rango = pd.date_range(start=fecha_min, end=fecha_max, freq="MS")

    df_formato = df[['POZO OFICIAL', 'YACIMIENTO']].drop_duplicates().assign(key=1)
    fechas_df = pd.DataFrame({'FECHA': fecha_rango, 'key': 1})

    return fechas_df.merge(df_formato, on='key').drop('key', axis=1).merge(df, on=["FECHA", "POZO OFICIAL", "YACIMIENTO"], how="left")

def calcular_promedio_general(df):  
    if df.empty:
        return pd.DataFrame()
    
    columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    return df.groupby('FECHA')[columnas_numericas].mean().reset_index()

def calcular_HI(df_pozos, df_promedio, hi_types):
    df_promedio = df_promedio.set_index("FECHA")

    def calc_HI(row, key):
        fecha = row['FECHA']
        if fecha in df_promedio.index and hi_types[key] in df_promedio.columns:
            promedio = df_promedio.loc[fecha, hi_types[key]]
            if promedio > 0:
                return (row[hi_types[key]] - promedio) / promedio
        return 0

    for key in hi_types:
        df_pozos[key] = df_pozos.apply(lambda row: calc_HI(row, key), axis=1)
        df_pozos['trend_' + key] = df_pozos.groupby('POZO OFICIAL')[key].cumsum()

    return df_pozos


def crear_figura_hi(df, x_col, y_col, titulo, eje_x, eje_y, q1, q2, q3, q4):
    fig = px.line(df, x=x_col, y=y_col, color="POZO OFICIAL", markers=True,
                  labels={x_col: eje_x, y_col: eje_y, "POZO OFICIAL": "Pozo"},
                  title=titulo)

    # Rango dinámico
    x_min, x_max = df[x_col].min(), df[x_col].max()
    y_min, y_max = df[y_col].min(), df[y_col].max()

    # Actualizar diseño del gráfico
    fig.update_layout(
        shapes=[
    dict(type="rect", xref="paper", yref="paper", x0=0, y0=0, x1=1, y1=1,
         line=dict(color="black", width=1), fillcolor='rgba(0,0,0,0)')
],
        plot_bgcolor="whitesmoke",
        xaxis=dict(
            title=eje_x,
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=1,
            showgrid=True,
            gridcolor='lightgray',
            tickfont=dict(color="black")
        ),
        yaxis=dict(
            title=eje_y,
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=1,
            showgrid=True,
            gridcolor='lightgray',
            tickfont=dict(color="black")
        )
    )

    # Anotaciones de cuadrantes
    coords = [(0.7, 0.7, q1), (0.1, 0.7, q2), (0.7, 0.1, q3), (0.1, 0.1, q4)]
    for xp, yp, label in coords:
        fig.add_annotation(
            x=x_min + xp * (x_max - x_min),
            y=y_min + yp * (y_max - y_min),
            text=label,
            showarrow=False,
            font=dict(size=16,color="brown")
        )

    return fig

def clasificar_pozos_por_cuadrante(df_hi, df_coor, x_col, y_col):
    # Obtener el último punto por pozo
    ultimos = df_hi.sort_values("FECHA").groupby("POZO OFICIAL").tail(1)

    def clasificar(row):
        x = row[x_col]
        y = row[y_col]
        if x >= 0 and y >= 0:
            return "Q1: RIESGO MODERADO"
        elif x < 0 and y >= 0:
            return "Q2: REQUIERE ATENCIÓN"
        elif x >= 0 and y < 0:
            return "Q3: ÓPTIMO"
        else:
            return "Q4: BAJO POTENCIAL"

    ultimos["CLASIFICACION"] = ultimos.apply(clasificar, axis=1)

    df_coords = df_coor.drop_duplicates(subset="POZO OFICIAL")
    resultado = ultimos.merge(df_coords, on="POZO OFICIAL", how="left")

    return resultado[["POZO OFICIAL", "XCOOR_OBJ", "YCOOR_OBJ", "CLASIFICACION"]]

def graficar_pozos_clasificados(df_clasificado, imagen_path, x_min, x_max, y_min, y_max):
    try:
        img = Image.open(requests.get(imagen_path, stream=True).raw if imagen_path.startswith("http") else imagen_path)
    except Exception as e:
        st.error(f"No se pudo cargar la imagen: {e}")
        return None

    fig = go.Figure()

    # Imagen de fondo
    fig.add_layout_image(
        dict(source=img, xref="x", yref="y",
             x=x_min, y=y_max,
             sizex=x_max - x_min, sizey=y_max - y_min,
             xanchor="left", yanchor="top", sizing="stretch", layer="below")
    )

    # # Colores por clasificación
    # colores = {
    #     "Q1: Alto / Alto": "red",
    #     "Q2: Bajo / Alto": "blue",
    #     "Q3: Alto / Bajo": "green",
    #     "Q4: Bajo / Bajo": "brown"
    # }
    
    # Colores por clasificación
    colores = {
        "Q1: RIESGO MODERADO": "red",
        "Q2: REQUIERE ATENCIÓN": "blue",
        "Q3: ÓPTIMO": "green",
        "Q4: BAJO POTENCIAL": "brown"
    }

    # Agregar pozos por categoría
    for clasificacion, df_group in df_clasificado.groupby("CLASIFICACION"):
        fig.add_trace(go.Scatter(
            x=df_group["XCOOR_OBJ"],
            y=df_group["YCOOR_OBJ"],
            mode='markers+text',
            marker=dict(size=15, color=colores.get(clasificacion)),
            text=df_group["POZO OFICIAL"],
            textfont=dict(size=12, color="black"),
            textposition="top center",
            name=clasificacion
        ))

    fig.update_layout(
        #title="Distribución de Pozos por Clasificación de Cuadrante",
        height=700,
        xaxis=dict(range=[x_min, x_max], title="XCOOR"),
        yaxis=dict(range=[y_min, y_max], title="YCOOR"),
        legend_title="Clasificación"
    )

    return fig



# ====================
# 4. CARGA Y FILTROS
# ====================

# Subir archivo
st.sidebar.markdown("### Cargar archivo CSV")
uploaded_file = st.sidebar.file_uploader("Selecciona un archivo CSV", type=["csv"])
if not uploaded_file:
    st.warning("⚠️ Carga un archivo CSV para comenzar.")
    st.stop()

data = cargar_datos(uploaded_file)

# Validar columnas
columnas_requeridas = [
    'POZO OFICIAL', 'FECHA', 'YACIMIENTO',
    'BRUTO DIARIO bpd','ACEITE DIARIO bpd','AGUA DIARIA bpd','GAS DIARIO pcd',
    'BRUTO ACUMULADO mbbl','ACEITE ACUMULADO mbbl','AGUA ACUMULADA mbbl','GAS ACUMULADO mmpc',
    'RPM'
]
faltantes = validar_columnas(data, columnas_requeridas)
if faltantes:
    st.error(f"❌ Faltan columnas requeridas: {faltantes}")
    st.stop()

df = filtrar_pozos_con_fecha(data[columnas_requeridas])
df_coor = data[['POZO OFICIAL', 'XCOOR_OBJ', 'YCOOR_OBJ']].dropna().drop_duplicates()

# Filtros interactivos
st.sidebar.header("Opciones de Filtro")
yacimientos = df['YACIMIENTO'].dropna().unique()
yacimientos_sel = st.sidebar.multiselect("Selecciona el yacimiento:", yacimientos)
if not yacimientos_sel:
    st.warning("⚠️ No has seleccionado un yacimiento.")
    st.stop()

df_yac = df[df['YACIMIENTO'].isin(yacimientos_sel)]
pozos = df_yac['POZO OFICIAL'].dropna().unique()
pozos_sel = st.sidebar.multiselect("Selecciona los pozos:", pozos)
if not pozos_sel:
    st.warning("⚠️ No has seleccionado un pozo.")
    st.stop()

df_pozos = df_yac[df_yac['POZO OFICIAL'].isin(pozos_sel)]

# ====================
# 5. PROCESAMIENTO Y GRÁFICOS
# ====================

hi_types = {
    'hi_bruta': 'BRUTO DIARIO bpd',
    'hi_oil': 'ACEITE DIARIO bpd',
    'hi_water': 'AGUA DIARIA bpd',
    'hi_gas': 'GAS DIARIO pcd',
    'hi_cum_bruta': 'BRUTO ACUMULADO mbbl',
    'hi_cum_oil': 'ACEITE ACUMULADO mbbl',
    'hi_cum_water': 'AGUA ACUMULADA mbbl',
    'hi_cum_gas': 'GAS ACUMULADO mmpc',
    'hi_rpm': 'RPM'
}

df_sec = secuencia_completa_fechas(df_pozos)
df_prom = calcular_promedio_general(df_sec)
df_hi = calcular_HI(df_pozos.copy(), df_prom, hi_types)

df_chart = df_hi[['POZO OFICIAL'] + [f"trend_{k}" for k in hi_types.keys()]].copy()

# ====================
# 6. VISUALIZACIONES
# ====================
tabs = st.tabs(["ÍNDICE HETEROGENEIDAD", 'MAPA'])

with tabs[0]:
    c1, c2, c3 = st.columns(3)

    with c1:
        st.plotly_chart(crear_figura_hi(df_chart, "trend_hi_oil", "trend_hi_bruta", "IH PRODUCCIÓN DIARIA: ACEITE & BRUTA",
                                        "IH ACEITE DIARIO", "IH BRUTA DIARIA", "Alto Qo & Alto Qb", "Bajo Qo & Alto Qb", "Alto Qo & Bajo Qb", "Bajo Qo & Bajo Qb"))
        
        st.plotly_chart(crear_figura_hi(df_chart, "trend_hi_cum_oil", "trend_hi_cum_bruta", "IH PRODUCCIÓN ACUMULADA: ACEITE & BRUTA",
                                        "IH ACEITE ACUMULADO", "IH BRUTA ACUMULADA", "Alto Np & Alto Nb", "Bajo Np & Alto Nb", "Alto Np & Bajo Nb", "Bajo Np & Bajo Nb"))
        
        st.plotly_chart(crear_figura_hi(df_chart, "trend_hi_rpm", "trend_hi_cum_oil", "IH RPM: RPM & ACEITE ACUMULADO",
                                        "RPM", "IH ACEITE ACUMULADO", "Alto RPM & Alto Np", "Bajo RPM & Alto Np", "Alto RPM & Bajo Np", "Bajo RPM & Bajo Np"))

    with c2:
        st.plotly_chart(crear_figura_hi(df_chart, "trend_hi_oil", "trend_hi_water", "IH PRODUCCIÓN DIARIA: ACEITE & AGUA",
                                        "IH ACEITE DIARIO", "IH AGUA DIARIA", "Alto Qo & Alto Qw", "Bajo Qo & Alto Qw", "Alto Qo & Bajo Qw", "Bajo Qo & Bajo Qw"))
        
        st.plotly_chart(crear_figura_hi(df_chart, "trend_hi_cum_oil", "trend_hi_cum_water", "IH PRODUCCIÓN ACUMULADA: ACEITE & AGUA",
                                        "IH ACEITE ACUMULADO", "IH AGUA ACUMULADA", "Alto Np & Alto Wp", "Bajo Np & Alto Wp", "Alto Np & Bajo Wp", "Bajo Np & Bajo Wp"))
        
        st.plotly_chart(crear_figura_hi(df_chart, "trend_hi_rpm", "trend_hi_cum_water", "IH RPM: RPM & AGUA ACUMULADA",
                                        "RPM", "IH AGUA ACUMULADA", "Alto RPM & Alto Wp", "Bajo RPM & Alto Wp", "Alto RPM & Bajo Wp", "Bajo RPM & Bajo Wp"))

    with c3:
        st.plotly_chart(crear_figura_hi(df_chart, "trend_hi_oil", "trend_hi_gas", "IH PRODUCCIÓN DIARIA: ACEITE & GAS",
                                        "IH ACEITE DIARIO", "IH GAS DIARIO", "Alto Qo & Alto Qg", "Bajo Qo & Alto Qg", "Alto Qo & Bajo Qg", "Bajo Qo & Bajo Qg"))
        
        st.plotly_chart(crear_figura_hi(df_chart, "trend_hi_cum_oil", "trend_hi_cum_gas", "IH PRODUCCIÓN ACUMULADA: ACEITE & GAS",
                                        "IH ACEITE ACUMULADO", "IH GAS ACUMULADA", "Alto Np & Alto Gp", "Bajo Np & Alto Gp", "Alto Np & Bajo Gp", "Bajo Np & Bajo Gp"))
        
        st.plotly_chart(crear_figura_hi(df_chart, "trend_hi_rpm", "trend_hi_cum_gas", "IH RPM: RPM & GAS ACUMULADO",
                                        "RPM", "IH GAS ACUMULADO", "Alto RPM & Alto Gp", "Bajo RPM & Alto Gp", "Alto RPM & Bajo Gp", "Bajo RPM & Bajo Gp"))
        
with tabs[1]:
    c1, c2, c3 = st.columns(3)
    with c1:
        
        # Invertir el diccionario para mostrar nombres descriptivos en el selectbox
        opciones = {v: k for k, v in hi_types.items()}
        
        x_label1 = st.selectbox("Eje X", list(opciones.keys()), index=1, key="x1")
        y_label2 = st.selectbox("Eje Y", list(opciones.keys()), index=0, key="y2")
        
        st.markdown(f"<h3 style='font-size:18px; color:#333;text-align:center;text-transform:uppercase;'>{x_label1} & {y_label2}</h3>", unsafe_allow_html=True)
        
        # Convertir a columnas del DataFrame
        x_col = f"trend_{opciones[x_label1]}"
        y_col = f"trend_{opciones[y_label2]}"
        
        #x_col = "trend_hi_oil"
        #y_col = "trend_hi_water"
        imagen_url = "https://raw.githubusercontent.com/carlosmr91/IH_ACE/main/poligono2.PNG"
        img_x_min, img_x_max = 543594, 588000
        img_y_min, img_y_max = 2440590.0, 2483940.0
    
        df_cuadrantes = clasificar_pozos_por_cuadrante(df_hi, df_coor, x_col, y_col)
        fig_mapa_cuadrantes_c1 = graficar_pozos_clasificados(df_cuadrantes, imagen_url, img_x_min, img_x_max, img_y_min, img_y_max)
        st.plotly_chart(fig_mapa_cuadrantes_c1, use_container_width=True, key='fig_c1')
        
    with c2:
        
        # Invertir el diccionario para mostrar nombres descriptivos en el selectbox
        opciones = {v: k for k, v in hi_types.items()}
        
        x_label3 = st.selectbox("Eje X", list(opciones.keys()), index=1, key="x3")
        y_label4 = st.selectbox("Eje Y", list(opciones.keys()), index=0, key="y4")
        
        st.markdown(f"<h3 style='font-size:18px; color:#333;text-align:center;text-transform:uppercase;'>{x_label3} & {y_label4}</h3>", unsafe_allow_html=True)

        
        # Convertir a columnas del DataFrame
        x_col = f"trend_{opciones[x_label3]}"
        y_col = f"trend_{opciones[y_label4]}"
        #x_col = "trend_hi_cum_oil"
        #y_col = "trend_hi_cum_water"
        imagen_url = "https://raw.githubusercontent.com/carlosmr91/IH_ACE/main/poligono2.PNG"
        img_x_min, img_x_max = 543594, 588000
        img_y_min, img_y_max = 2440590.0, 2483940.0
    
        df_cuadrantes = clasificar_pozos_por_cuadrante(df_hi, df_coor, x_col, y_col)
        fig_mapa_cuadrantes_c2 = graficar_pozos_clasificados(df_cuadrantes, imagen_url, img_x_min, img_x_max, img_y_min, img_y_max)
        st.plotly_chart(fig_mapa_cuadrantes_c2, use_container_width=True, key='fig_c2')
        
    with c3:
        
        # Invertir el diccionario para mostrar nombres descriptivos en el selectbox
        opciones = {v: k for k, v in hi_types.items()}
                
        x_label5 = st.selectbox("Eje X", list(opciones.keys()), index=1, key="x5")
        y_label6 = st.selectbox("Eje Y", list(opciones.keys()), index=0, key="x6")
        
        st.markdown(f"<h3 style='font-size:18px; color:#333;text-align:center;text-transform:uppercase;'>{x_label5} & {y_label6}</h3>", unsafe_allow_html=True)
        
        # Convertir a columnas del DataFrame
        x_col = f"trend_{opciones[x_label5]}"
        y_col = f"trend_{opciones[y_label6]}"
        #x_col = "trend_hi_oil"
        #y_col = "trend_hi_gas"
        imagen_url = "https://raw.githubusercontent.com/carlosmr91/ACE_IH_PROD/main/poligono2.PNG"
        img_x_min, img_x_max = 543594, 588000
        img_y_min, img_y_max = 2440590.0, 2483940.0
    
        df_cuadrantes = clasificar_pozos_por_cuadrante(df_hi, df_coor, x_col, y_col)
        fig_mapa_cuadrantes_c3 = graficar_pozos_clasificados(df_cuadrantes, imagen_url, img_x_min, img_x_max, img_y_min, img_y_max)
        st.plotly_chart(fig_mapa_cuadrantes_c3, use_container_width=True, key='fig_c3')    
       


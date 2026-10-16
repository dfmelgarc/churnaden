import streamlit as st

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from statsmodels.tsa.holtwinters import ExponentialSmoothing

import io

import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Sistema IA - Ventas y Retención", layout="wide", page_icon="🚀")

# 1. Función para descargar el dataframe como Excel real

def convertir_df_a_excel(df):

    output = io.BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:

        df.to_excel(writer, index=False, sheet_name='Datos')

    return output.getvalue()

st.title("Sistema Inteligente: Operaciones y Comercial")

st.markdown("Sube tu Excel maestro con el historial de ventas.")

archivo_subido = st.sidebar.file_uploader("Sube tu archivo Excel (.xlsx)", type=["xlsx"])

if archivo_subido is not None:

    try:

        df_raw = pd.read_excel(archivo_subido)

        if 'Fecha' in df_raw.columns:

            df_raw['Fecha'] = pd.to_datetime(df_raw['Fecha'], errors='coerce')

        else:

            st.error("❌ El Excel debe contener una columna llamada 'Fecha'.")

            st.stop()

            

        tab1, tab2 = st.tabs(["👥 Sistema Anti-Churn (Clientes)", "📈 Análisis y Predicción de Inventario"])

        

        # ==========================================

        # PESTAÑA 1: SISTEMA ANTI-CHURN

        # ==========================================

        with tab1:

            st.header("🎯 Análisis de Retención y Fuga de Clientes")

            cols_churn = ['ID_Cliente', 'Nombre_Cliente', 'Fecha', 'Monto', 'Categoria']

            if all(col in df_raw.columns for col in cols_churn):

                max_date = df_raw['Fecha'].max()

                

                # RFM Básico

                df_rfm = df_raw.groupby('ID_Cliente').agg(

                    Nombre_Cliente=('Nombre_Cliente', 'first'),

                    Recencia=('Fecha', lambda x: (max_date - x.max()).days),

                    Frecuencia=('Fecha', 'count'),

                    Valor_Monetario=('Monto', 'sum')

                ).reset_index()

                

                # 2. Agregar las últimas 3 compras de cada cliente

                ultimos_productos = df_raw.sort_values('Fecha').groupby('ID_Cliente')['Categoria'].apply(

                    lambda x: ' | '.join(x.tail(3).astype(str).tolist())

                ).reset_index()

                ultimos_productos.rename(columns={'Categoria': 'Ultimas_3_Compras'}, inplace=True)

                

                # Unir el historial al RFM

                df_rfm = df_rfm.merge(ultimos_productos, on='ID_Cliente', how='left')

                

                df_perdidos = df_rfm[df_rfm['Recencia'] > 365].copy()

                df_perdidos['Semaforo_Riesgo'] = 'Recuperación'

                

                df_activos = df_rfm[df_rfm['Recencia'] <= 365].copy()

                df_activos['Churn_Target'] = np.where(df_activos['Recencia'] > 180, 1, 0)

                

                if len(df_activos) > 5:

                    X = df_activos[['Frecuencia', 'Valor_Monetario']]

                    y = df_activos['Churn_Target']

                    model = RandomForestClassifier(max_depth=5, random_state=42)

                    model.fit(X, y)

                    

                    probs = model.predict_proba(X)[:, 1]

                    df_activos['Semaforo_Riesgo'] = np.where(probs >= 0.40, 'Riesgo Alto', 'OK')

                    

                    llamadas_urgentes = df_activos[df_activos['Semaforo_Riesgo'] == 'Riesgo Alto'].sort_values('Valor_Monetario', ascending=False)

                    ya_perdidos = df_perdidos.sort_values('Valor_Monetario', ascending=False)

                    clientes_ok = df_activos[df_activos['Semaforo_Riesgo'] == 'OK'].sort_values('Recencia')

                    

                    columnas_mostrar = ['Nombre_Cliente', 'Recencia', 'Valor_Monetario', 'Ultimas_3_Compras']

                    

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.error(f"🔴 Para Llamar Hoy ({len(llamadas_urgentes)})")

                        st.dataframe(llamadas_urgentes[columnas_mostrar], hide_index=True)

                        st.download_button("📥 Bajar en Excel", convertir_df_a_excel(llamadas_urgentes[columnas_mostrar]), "llamar_hoy.xlsx", key="b1")

                    with col2:

                        st.warning(f"🟠 Ya se fueron ({len(ya_perdidos)})")

                        st.dataframe(ya_perdidos[columnas_mostrar], hide_index=True)

                        st.download_button("📥 Bajar en Excel", convertir_df_a_excel(ya_perdidos[columnas_mostrar]), "perdidos.xlsx", key="b2")

                    with col3:

                        st.success(f"🟢 Están OK ({len(clientes_ok)})")

                        st.dataframe(clientes_ok[columnas_mostrar], hide_index=True)

                        st.download_button("📥 Bajar en Excel", convertir_df_a_excel(clientes_ok[columnas_mostrar]), "clientes_ok.xlsx", key="b3")

                else:

                    st.warning("Faltan datos para entrenar el modelo.")

            else:

                st.error("Faltan columnas necesarias para el Churn.")

        # ==========================================

        # PESTAÑA 2: PREDICCIÓN DE INVENTARIOS

        # ==========================================

        with tab2:

            st.header("📦 Comportamiento de Inventario y Estacionalidad")

            if 'Categoria' in df_raw.columns and 'Cantidad' in df_raw.columns:

                

                # 3. Eliminar lógica de contenedores y enfocar en Top Categorías y estacionalidad

                st.subheader("Top Categorías Históricas (Lo que más se vende en total)")

                top_cat = df_raw.groupby('Categoria')['Cantidad'].sum().sort_values(ascending=False).head(10).reset_index()

                st.dataframe(top_cat, use_container_width=True)

                

                df_categoria_mes = df_raw.groupby([pd.Grouper(key='Fecha', freq='ME'), 'Categoria'])['Cantidad'].sum().reset_index()

                df_series = df_categoria_mes.pivot(index='Fecha', columns='Categoria', values='Cantidad').fillna(0)

                

                if len(df_series) >= 24:

                    st.subheader("Predicción de Demanda Mensual (Próximos 12 meses)")

                    fechas_futuras = pd.date_range(start=df_series.index[-1] + pd.Timedelta(days=1), periods=12, freq='ME')

                    plan_maestro = pd.DataFrame({'Mes': fechas_futuras.strftime('%Y-%m')})

                    

                    for cat in df_series.columns:

                        if df_series[cat].sum() > 0:

                            try:

                                # Uso de 'add' en lugar de 'mul' para evitar problemas con ceros, y clipping para evitar negativos

                                modelo_cat = ExponentialSmoothing(

                                    df_series[cat], trend='add', seasonal='add', seasonal_periods=12, initialization_method="estimated"

                                ).fit()

                                # 4. Corrección de los negativos (np.maximum fuerza que cualquier negativo se vuelva 0)

                                prediccion = np.maximum(0, np.round(modelo_cat.forecast(12).values))

                                plan_maestro[cat] = prediccion.astype(int)

                            except:

                                plan_maestro[cat] = 0

                                

                    st.dataframe(plan_maestro, use_container_width=True)

                    st.download_button("📥 Descargar Proyección Excel", convertir_df_a_excel(plan_maestro), "proyeccion_inventario.xlsx", key="b4")

                    

                    st.subheader("Visualización de Estacionalidad Histórica")

                    st.line_chart(df_series)

                else:

                    st.warning("Se necesitan al menos 24 meses de historial para predecir estacionalidad mensual de forma matemática.")

            else:

                st.error("Faltan columnas de Categoria y Cantidad para el inventario.")

    except Exception as e:

        st.error(f"Error procesando el archivo Excel: {e}")

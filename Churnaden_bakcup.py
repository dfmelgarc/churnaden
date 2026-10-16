import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings("ignore")

# 1. Configuración de la página
st.set_page_config(page_title="Sistema IA - Ventas y Retención", layout="wide", page_icon="🚀")

st.title("Sistema Inteligente: Operaciones y Comercial")
st.markdown("Sube tu Excel maestro con el historial de ventas. El sistema usa IA para predecir tus compras por contenedor y detectar clientes en riesgo de abandono.")

# 2. Carga del archivo en la barra lateral
st.sidebar.header("📂 Carga de Datos")
archivo_subido = st.sidebar.file_uploader("Sube tu archivo Excel (.xlsx)", type=["xlsx"])

# Lógica del contenedor (input dinámico)
st.sidebar.markdown("---")
st.sidebar.header("📦 Configuración de Logística")
capacidad_contenedor = st.sidebar.number_input("Capacidad por Contenedor (Ej. fardos/pacas):", min_value=1, value=400)

if archivo_subido is not None:
    try:
        # Leer el archivo Excel cargado por el usuario
        df_raw = pd.read_excel(archivo_subido)
        
        # Limpieza básica: asegurar que la columna Fecha sea formato tiempo
        if 'Fecha' in df_raw.columns:
            df_raw['Fecha'] = pd.to_datetime(df_raw['Fecha'], errors='coerce')
        else:
            st.error("❌ El Excel debe contener una columna llamada 'Fecha'.")
            st.stop()
            
        st.success("✅ Datos cargados correctamente. Explora los resultados en las pestañas inferiores.")
        
        # Crear pestañas para la interfaz
        tab1, tab2 = st.tabs(["👥 Sistema Anti-Churn (Clientes)", "📈 Predicción de Inventario (Compras)"])
        
        # ==========================================
        # PESTAÑA 1: SISTEMA ANTI-CHURN
        # ==========================================
        with tab1:
            st.header("🎯 Análisis de Retención y Fuga de Clientes")
            
            # Verificar que existan las columnas para el modelo B
            cols_churn = ['ID_Cliente', 'Nombre_Cliente', 'No_Documento', 'Monto']
            if all(col in df_raw.columns for col in cols_churn):
                max_date = df_raw['Fecha'].max()
                
                # 1. Ingeniería de Características (RFM)
                df_rfm = df_raw.groupby('ID_Cliente').agg(
                    Nombre_Cliente=('Nombre_Cliente', 'first'),
                    Recencia=('Fecha', lambda x: (max_date - x.max()).days),
                    Frecuencia=('No_Documento', 'count'),
                    Valor_Monetario=('Monto', 'sum')
                ).reset_index()
                
                # 2. Segmentación (Clientes perdidos vs Activos)
                df_perdidos = df_rfm[df_rfm['Recencia'] > 365].copy()
                df_perdidos['Semaforo_Riesgo'] = 'Recuperación (Ya se fueron)'
                
                df_activos = df_rfm[df_rfm['Recencia'] <= 365].copy()
                df_activos['Churn_Target'] = np.where(df_activos['Recencia'] > 180, 1, 0)
                
                # 3. Entrenamiento del Modelo Random Forest
                X = df_activos[['Frecuencia', 'Valor_Monetario']]
                y = df_activos['Churn_Target']
                
                if len(df_activos) > 5: # Validar que haya datos
                    model = RandomForestClassifier(max_depth=5, random_state=42)
                    model.fit(X, y)
                    
                    # 4. Predicción y asignación de semáforo a la cartera activa
                    probs = model.predict_proba(X)[:, 1]
                    df_activos['Semaforo_Riesgo'] = np.where(probs >= 0.40, 'Para Llamar Hoy (Riesgo Alto)', 'Están OK (Bajo Riesgo)')
                    
                    # 5. Despliegue de Resultados (Listas de Acción)
                    col1, col2, col3 = st.columns(3)
                    
                    llamadas_urgentes = df_activos[df_activos['Semaforo_Riesgo'] == 'Para Llamar Hoy (Riesgo Alto)']
                    ya_perdidos = df_perdidos
                    clientes_ok = df_activos[df_activos['Semaforo_Riesgo'] == 'Están OK (Bajo Riesgo)']
                    
                    with col1:
                        st.error(f"🔴 Para Llamar Hoy ({len(llamadas_urgentes)})")
                        st.write("Clientes que podrían irse. ¡Prioridad de contacto!")
                        st.dataframe(llamadas_urgentes[['Nombre_Cliente', 'Recencia', 'Valor_Monetario']].sort_values('Valor_Monetario', ascending=False), hide_index=True)
                    
                    with col2:
                        st.warning(f"🟠 Recuperación ({len(ya_perdidos)})")
                        st.write("Clientes que ya se fueron (> 1 año sin compra).")
                        st.dataframe(ya_perdidos[['Nombre_Cliente', 'Recencia', 'Valor_Monetario']].sort_values('Valor_Monetario', ascending=False), hide_index=True)
                        
                    with col3:
                        st.success(f"🟢 Están OK ({len(clientes_ok)})")
                        st.write("Compran de manera constante y natural.")
                        st.dataframe(clientes_ok[['Nombre_Cliente', 'Recencia', 'Valor_Monetario']].sort_values('Recencia'), hide_index=True)
                        
                else:
                    st.warning("No hay suficientes datos de clientes para entrenar el modelo.")
            else:
                st.error(f"Faltan columnas. El sistema Churn necesita: {cols_churn}")

        # ==========================================
        # PESTAÑA 2: PREDICCIÓN DE INVENTARIOS
        # ==========================================
        with tab2:
            st.header("🚢 Proyección de Compras y Contenedores")
            
            # Verificar columnas para el modelo A
            cols_inv = ['Categoria', 'Cantidad']
            if all(col in df_raw.columns for col in cols_inv):
                
                # 1. Agrupación mensual
                df_categoria_mes = df_raw.groupby([pd.Grouper(key='Fecha', freq='ME'), 'Categoria'])['Cantidad'].sum().reset_index()
                df_series = df_categoria_mes.pivot(index='Fecha', columns='Categoria', values='Cantidad').fillna(1)
                
                if len(df_series) >= 24: # Se requieren al menos 24 meses para captar estacionalidad
                    st.subheader("Plan Maestro de Compras (Próximos 12 meses)")
                    
                    # Generar fechas futuras
                    fechas_futuras = pd.date_range(start=df_series.index[-1] + pd.Timedelta(days=1), periods=12, freq='ME')
                    plan_maestro = pd.DataFrame({'Mes': fechas_futuras.strftime('%Y-%m')})
                    
                    # 2. Bucle de modelación Holt-Winters por categoría
                    for cat in df_series.columns:
                        try:
                            modelo_cat = ExponentialSmoothing(
                                df_series[cat], trend='add', seasonal='mul', seasonal_periods=12
                            ).fit()
                            prediccion = modelo_cat.forecast(12)
                            plan_maestro[cat] = np.round(prediccion.values).astype(int)
                        except Exception:
                            # Respaldo si una categoría falla matemáticamente
                            plan_maestro[cat] = 0 
                            
                    # 3. Lógica Logística (Contenedores)
                    categorias_proyectadas = [c for c in plan_maestro.columns if c != 'Mes']
                    plan_maestro['Total_Unidades_Proyectadas'] = plan_maestro[categorias_proyectadas].sum(axis=1)
                    plan_maestro['Contenedores_Sugeridos'] = np.ceil(plan_maestro['Total_Unidades_Proyectadas'] / capacidad_contenedor).astype(int)
                    
                    # Resaltar datos visualmente
                    st.dataframe(plan_maestro.style.background_gradient(cmap='Blues', subset=['Total_Unidades_Proyectadas']), use_container_width=True)
                    
                    # 4. Gráfico de Tendencias
                    st.subheader("Visualización de Tendencia Histórica")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    for cat in df_series.columns:
                        ax.plot(df_series.index, df_series[cat], label=cat, marker='.')
                    ax.set_ylabel("Cantidad Vendida")
                    ax.legend(title="Categorías", bbox_to_anchor=(1.05, 1), loc='upper left')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                else:
                    st.warning("⚠️ El modelo Holt-Winters de predicción necesita al menos 24 meses de historial para entender la estacionalidad (temporadas altas/bajas) anual de Guatemala. Sube un archivo con más historia.")
            else:
                st.error(f"Faltan columnas. El sistema de Inventario necesita: {cols_inv}")

    except Exception as e:
        st.error(f"Ocurrió un error al procesar el archivo Excel: {str(e)}")

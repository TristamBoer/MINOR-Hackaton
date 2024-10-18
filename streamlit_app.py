import json
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import plotly_express as px


def page_config():
    st.set_page_config(layout='wide')


page_config()


st.title('Dashboard Hackaton Data Science 2024')
st.write(
    '''
    Welkom bij de dashboard van groep 2 voor de Hackaton van de Minor Data Science 2024. 
    De Hackaton gaat de overgang naar duurzame energie in de Nederlandse logistieke sector.
    Een belangrijk onderdeel hiervan is de overstap naar emissie-vrije mobiliteit.
    Daarvoor is ons te taak gegeven om een bijdrage te leveren aan dit doel door twee bedrijventerreinen een overzicht te maken van alle factoren die energie vragen.
    De dashboard is gericht op het centraliseren van verschillende online vindbare datasets.
    Hierbij is er gekeken naar de enerievraag van de panden, aantal werknemers per bedrijf en de aansluitingen van het netbeheer.
    '''
)


# https://app.bedrijvenopdekaart.nl/?bodkdata=N4IgZiBcDaIIYBsEgLoBoQCMrWgFgDoB2ANgE4BGNAVgCYC8AGEvdfAgDgoGYyb6mLNoQ6M8VOgW4cSJYcSIc+k6bJQoAvkA
@st.cache_data
def eigen_excel_file(sheet):
    return pd.read_excel('MINOR - Hackaton - werknemer aantal.xlsx', sheet)

df_werknemers = eigen_excel_file('Werknemers')
df_werknemers['Aantal'] = round(((df_werknemers['Aantal min'] + df_werknemers['Aantal max']) / 2), 0).astype(int)

st.title('Aantal bekende werknemers per bedrijf') 
st.write(
    '''
    Door een online map gemaakt door 'Bedrijven Op Kaart', kan een schatting worden gemaakt voor de aantal werknemers per bedrijf.
    De cijfers in het figuur is gebaseerd op het afgeronde gemiddelde van de schatting. 
    Van niet alle bedrijven is de data bekend.
    Als dit het geval is, wordt het bedrijf niet mee genomen in het figuur.
    '''
)

fig = go.Figure()

fig.add_trace(go.Bar(
    x=df_werknemers['Bedrijf'],
    y=df_werknemers['Aantal']
))

fig.update_layout(
    title='Aantal werknemers per bedrijf in Sloterdijk Poort Noord',
    xaxis_title='Bedrijf',
    yaxis_title='Aantal werknemers'
)

st.plotly_chart(fig)



employeedata = eigen_excel_file('Werknemers')

st.title("Sloterdijk Poort Noord Employee Data")

labels = employeedata['Bedrijf']
max_values = employeedata['Aantal max']
min_values = employeedata['Aantal min']
sectors = employeedata['Sectoren']

MinMax = st.selectbox("Minimum Estimated or Maximum Estimated employees", ("Minimum", "Maximum"))

def autopct_format(values):
    total = sum(values)
    return [f'{v / total * 100:.1f}%' if v / total >= 0.02 else '' for v in values]

if MinMax == "Maximum":
    pie_fig = px.pie(
        names=labels,
        values=max_values,
        title='Distribution of Max Employees by Company',
        labels={'names': 'Company'},
        color_discrete_sequence=px.colors.qualitative.T10,
    )
    pie_fig.update_traces(textinfo='label+percent', pull=[0.1 if value/sum(max_values) > 0.02 else 0 for value in max_values])

elif MinMax == "Minimum":
    pie_fig = px.pie(
        names=labels,
        values=min_values,
        title='Distribution of Min Employees by Company',
        labels={'names': 'Company'},
        color_discrete_sequence=px.colors.qualitative.T10,
    )
    pie_fig.update_traces(textinfo='label+percent', pull=[0.1 if value/sum(min_values) > 0.02 else 0 for value in min_values])

st.plotly_chart(pie_fig)




sector_counts = employeedata['Sector'].value_counts()

sector_fig = px.pie(
    names=sector_counts.index,
    values=sector_counts.values,
    title='Distribution of Companies by Sector',
    labels={'names': 'Sector'},
    color_discrete_sequence=px.colors.qualitative.T10,
)

# Update the traces to display percentages
sector_fig.update_traces(textinfo='label+percent')

# Display the pie chart in Streamlit
st.plotly_chart(sector_fig)




st.title('Voorspelling gas & stroom gebruik tot 2050')


@st.cache_data
def verbruik_excel():
    return pd.read_excel('verbruik_per_sector.xlsx', sheet_name='Sheet2')


df = verbruik_excel()

options = {
    'Nationaal verbruik': ('Electricity_Consumption_TWh', 'Gas_Consumption_Bcm'),
    'Sloterdijk Poort-Noord': ('kWh Sloterdijk Poort-Noord', 'M gas Sloterdijk Poort-Noord'),
    'Dutch fresh port': ('kWh dutch fresh port', 'm3 gas dutch fresh port')
}

selected_option = st.radio(
    "Selecteer de weergave:",
    list(options.keys()),
    horizontal=True
)

electricity_option, gas_option = options[selected_option]

X = df['Year'].values.reshape(-1, 1)
y_electricity = df[electricity_option].values
model_electricity = LinearRegression()
model_electricity.fit(X, y_electricity)

future_years = np.arange(2022, 2051).reshape(-1, 1)
predicted_electricity = model_electricity.predict(future_years) * 1.02  # 2% groei voor elektriciteit

electricity_fig = go.Figure()

electricity_fig.add_trace(go.Scatter(
    x=df['Year'], 
    y=df[electricity_option],
    mode='lines',
    name=f'Historisch {electricity_option}',
    line=dict(color='blue')
))

electricity_fig.add_trace(go.Scatter(
    x=future_years.flatten(),
    y=predicted_electricity,
    mode='lines',
    name=f'Voorspeld {electricity_option}',
    line=dict(color='green', dash='dash')
))

electricity_fig.update_layout(
    title=f'{electricity_option} consumptievoorspelling (2015-2050)',
    xaxis_title='Jaar',
    yaxis_title=electricity_option,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

y_gas = df[gas_option].values
model_gas = LinearRegression()
model_gas.fit(X, y_gas)

predicted_gas = model_gas.predict(future_years) * 0.98  # 2% afname voor gas

gas_fig = go.Figure()

gas_fig.add_trace(go.Scatter(
    x=df['Year'], 
    y=df[gas_option],
    mode='lines',
    name=f'Historisch {gas_option}',
    line=dict(color='red')
))

gas_fig.add_trace(go.Scatter(
    x=future_years.flatten(),
    y=predicted_gas,
    mode='lines',
    name=f'Voorspeld {gas_option}',
    line=dict(color='orange', dash='dash')
))

gas_fig.update_layout(
    title=f'{gas_option} consumptievoorspelling (2015-2050)',
    xaxis_title='Jaar',
    yaxis_title=gas_option,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(electricity_fig, use_container_width=True)

with col2:
    st.plotly_chart(gas_fig, use_container_width=True)

st.markdown("<p style='text-align: center;'>Bron: CBS - Levering aardgas, elektriciteit via openbaar net; bedrijven, SBI2008, regio</p>", unsafe_allow_html=True)




@st.cache_data
def table_csv():
    return pd.read_csv('table__83374NED.csv')

data = table_csv()

# Vervang komma's door punten en converteer naar numeriek, foutieve waarden worden NaN
data['Gemiddeld aardgasverbruik (m3/m2)'] = pd.to_numeric(data['Gemiddeld aardgasverbruik (m3/m2)'].str.replace(',', '.'), errors='coerce')
data['Gemiddeld elektriciteitsverbruik (kWh/m2)'] = pd.to_numeric(data['Gemiddeld elektriciteitsverbruik (kWh/m2)'].str.replace(',', '.'), errors='coerce')

# Verwijder rijen met NaN-waarden in de relevante kolommen
data = data.dropna(subset=['Gemiddeld aardgasverbruik (m3/m2)', 'Gemiddeld elektriciteitsverbruik (kWh/m2)'])

# Specifieke sectoren filteren
sectoren = [
    "Autobedrijf: autoschadeherstelbedrijven",
    "Autobedrijf: showroom en garage",
    "Detailhandel met koeling",
    "Detailhandel zonder koeling",
    "Groothandel met koeling",
    "Groothandel zonder koeling",
    "Kantoor: overheid",
    "Kantoor: overig"
]

# Filter de data op de specifieke sectoren
filtered_data = data[data['Utiliteitsbouw dienstensector'].isin(sectoren)]

# Groepeer de gefilterde data op 'Utiliteitsbouw dienstensector' en bereken het gemiddelde
gemiddeld_verbruik = filtered_data.groupby('Utiliteitsbouw dienstensector')[['Gemiddeld aardgasverbruik (m3/m2)', 'Gemiddeld elektriciteitsverbruik (kWh/m2)']].mean().reset_index()

# Streamlit layout
st.title("Totaal Aardgas- en Elektriciteitsverbruik per Sector")

# Slider voor de oppervlakte (m²)
oppervlakte = st.slider(
    "Selecteer de oppervlakte (m²):",
    min_value=100,
    max_value=5000,
    step=100,
    value=1000  # Standaard waarde
)

# Checkboxes voor het selecteren van sectoren
geselecteerde_sectoren = st.multiselect(
    "Selecteer de sectoren:",
    options=gemiddeld_verbruik['Utiliteitsbouw dienstensector'].tolist(),
    default=sectoren
)

# Filter de data op de geselecteerde sectoren
filtered_verbruik = gemiddeld_verbruik[gemiddeld_verbruik['Utiliteitsbouw dienstensector'].isin(geselecteerde_sectoren)]

# Bereken het totale verbruik op basis van de opgegeven oppervlakte
filtered_verbruik['Totaal aardgasverbruik (m3)'] = filtered_verbruik['Gemiddeld aardgasverbruik (m3/m2)'] * oppervlakte
filtered_verbruik['Totaal elektriciteitsverbruik (kWh)'] = filtered_verbruik['Gemiddeld elektriciteitsverbruik (kWh/m2)'] * oppervlakte

# Aardgas omrekenen naar elektrische energie equivalent
filtered_verbruik['Aardgas omgezet naar elektriciteit (kWh)'] = filtered_verbruik['Totaal aardgasverbruik (m3)'] * 9.769

# Maak de barplots voor gas- en elektriciteitsverbruik
fig = go.Figure()

# Barplot voor totaal gasverbruik
fig.add_trace(go.Bar(
    x=filtered_verbruik['Utiliteitsbouw dienstensector'],
    y=filtered_verbruik['Totaal aardgasverbruik (m3)'],
    name='Totaal Gasverbruik (m³)',
    marker_color='blue'
))

# Barplot voor totaal elektriciteitsverbruik
fig.add_trace(go.Bar(
    x=filtered_verbruik['Utiliteitsbouw dienstensector'],
    y=filtered_verbruik['Totaal elektriciteitsverbruik (kWh)'],
    name='Totaal Elektriciteitsverbruik (kWh)',
    marker_color='orange'
))

# Barplot voor het omrekenen van aardgas naar elektriciteit
fig.add_trace(go.Bar(
    x=filtered_verbruik['Utiliteitsbouw dienstensector'],
    y=filtered_verbruik['Aardgas omgezet naar elektriciteit (kWh)'],
    name='Aardgas omgezet naar elektriciteit',
    marker_color='green'
))

st.plotly_chart(fig)




# JSON-bestand inladen
with open('zonev.json', 'r') as f:
    data = json.load(f)

# Zet het JSON-bestand om naar een DataFrame
df = pd.DataFrame(data)

# Datumkolom converteren naar datetime-formaat
df['date'] = pd.to_datetime(df['date'])

# Data aggregeren voor dagelijkse, wekelijkse en maandelijkse opbrengst
df['day'] = df['date'].dt.floor('D')
df['week'] = df['date'].dt.to_period('W').apply(lambda r: r.start_time)
df['month'] = df['date'].dt.to_period('M').apply(lambda r: r.start_time)

# Aannames voor zonnepaneel (300PV)
panel_capacity = 300  # vermogen in watt
df['energy'] = (df['Q'] * panel_capacity) / 1000  # omzetten naar kWh

# Dagelijkse, wekelijkse en maandelijkse som van opgewekte energie
df_daily = df.groupby('day')['energy'].sum().reset_index()
df_weekly = df.groupby('week')['energy'].sum().reset_index()
df_monthly = df.groupby('month')['energy'].sum().reset_index()

# Streamlit interface
st.title('Opbrengst van Zonnepanelen')

st.write(
    '''
    Dit overzicht toont wanneer de zonnepanelen optimaal werken en wanneer ze minder efficiënt zijn. 
    Deze analyse is gebaseerd op data van het KNMI-station in De Bilt, waarbij de temperatuur (T) en de zonnestraling (Q) zijn gebruikt om het energieopbrengstpotentieel van de zonnepanelen te berekenen.
    Zonnepanelen werken het best bij een combinatie van gunstige zonnestraling en gematigde temperaturen. In dit model heb ik berekend hoe deze factoren samenhangen met de prestaties van zonnepanelen. 
    Wanneer de zonnestraling (Q) hoog is en de temperatuur binnen een geschikt bereik ligt, neemt de energieproductie toe. Bij extreme hitte of bewolkte dagen, wanneer de zonnestraling laag is, daalt de efficiëntie.
    Deze inzichten kunnen helpen om te bepalen wanneer zonnepanelen het meeste rendement opleveren, wat handig is voor het optimaliseren van energiebeheer en investeringen in zonne-energie.
    '''
)

# Selectiebox voor de frequentie
frequentie = st.selectbox(
    'Selecteer de frequentie van de opbrengstweergave:',
    ['Dagelijks', 'Wekelijks', 'Maandelijks']
)

# Maak de figuur
fig = go.Figure()

if frequentie == 'Dagelijks':
    # Voeg de dagelijkse data toe
    fig.add_trace(go.Scatter(x=df_daily['day'], y=df_daily['energy'], mode='lines+markers', name='Dagelijkse Opbrengst'))
elif frequentie == 'Wekelijks':
    # Voeg de wekelijkse data toe
    fig.add_trace(go.Scatter(x=df_weekly['week'], y=df_weekly['energy'], mode='lines+markers', name='Wekelijkse Opbrengst'))
elif frequentie == 'Maandelijks':
    # Voeg de maandelijkse data toe
    fig.add_trace(go.Scatter(x=df_monthly['month'], y=df_monthly['energy'], mode='lines+markers', name='Maandelijkse Opbrengst'))

# Update de layout van de grafiek
fig.update_layout(
    title=f'Opbrengst van Zonnepanelen ({frequentie})',
    xaxis_title='Datum',
    yaxis_title='Opgewekte Energie [kWh]'
)

# Toon de plot
st.plotly_chart(fig)


with open('zonev.json', 'r') as f:
    data = json.load(f)

# Zet het JSON-bestand om naar een DataFrame
df = pd.DataFrame(data)

# Datumkolom converteren naar datetime-formaat
df['date'] = pd.to_datetime(df['date'])

# Neem aan dat elk paneel een vermogen van 300 W heeft
panel_capacity = 300  # vermogen in watt

# Aantal zonnepanelen voor verschillende opties
panel_options = [12000, 13000, 14000, 15000]

# Streamlit interface
st.title('Opbrengst van Zonnepanelen in Sloterdijk Noord Poort')

st.write(
    '''
    Deze grafiek toont de energieopbrengst van de huidige 12.000 zonnepanelen op het terrein en geeft 
    inzicht in hoeveel extra energie gegenereerd kan worden bij een eventuele opschaling van het aantal zonnepanelen.
    Door verschillende scenario’s te vergelijken (bijvoorbeeld 13.000, 14.000 en 15.000 panelen), 
    wordt zichtbaar hoeveel meer energie kan worden opgewekt naarmate het aantal zonnepanelen toeneemt. 
    Dit kan helpen bij strategische beslissingen voor uitbreiding, omdat het laat zien welk rendement te verwachten is na een vergroting van de capaciteit.
    '''
)

# Selectiebox voor het aantal zonnepanelen
selected_panels = st.selectbox(
    'Selecteer het aantal zonnepanelen:',
    panel_options
)

# Bereken de totale energieopbrengst voor het geselecteerde aantal zonnepanelen
df['energy'] = (df['Q'] * panel_capacity * selected_panels) / 1000  # omzetten naar kWh

# Dagelijkse opbrengst
df_daily = df.groupby(df['date'].dt.date)['energy'].sum().reset_index()

# Maak de figuur
fig = go.Figure()

# Voeg de dagelijkse data toe aan de figuur
fig.add_trace(go.Scatter(
    x=df_daily['date'],
    y=df_daily['energy'],
    mode='lines+markers',
    name=f'Opbrengst met {selected_panels} Panelen'
))

# Update de layout van de grafiek
fig.update_layout(
    title=f'Opbrengst van Zonnepanelen met {selected_panels} Panelen',
    xaxis_title='Datum',
    yaxis_title='Opgewekte Energie [kWh]'
)

# Toon de plot in Streamlit
st.plotly_chart(fig)



@st.cache_data
def zonnepanelen_csv():
    return pd.read_csv('ZONNEPANELEN.csv', delimiter=';')


df = zonnepanelen_csv()
df_1 = df[
    (df['Gebruiksdoel'] == 'industriefunctie') |
    (df['Gebruiksdoel'] == 'logiesfunctie') |
    (df['Gebruiksdoel'] == 'kantoorfunctie')
][['nl2023_panelen.1', 'nl2023_wp', 'LNG', 'LAT']]

df_2 = df_1[
    (df_1['LAT'] <= 52.398259) & (df_1['LAT'] >= 52.393500) &
    (df_1['LNG'] <= 4.803099) & (df_1['LNG'] >= 4.778193)
]


st.title('Aantal zonnepanelen per bedrijf')
st.write(
    '''
    De gemeente van Amsterdam bevat informatie over de hoeveelheid zonnepanelen in van 2016 tot 2023. 
    De folium map hieronder maakt gebruikt van de data in het jaar 2023. 
    Grotendeels van het industrieterrein bevatte in 2023 nog geen zonnepanelen. 
    De grote van cirkels wordt bepaald door de hoeveelheid zonnepanelen. 
    Dus een bedrijf met meer zonnepanelen, heeft een grotere cirkel.  
    '''
)


def aantal_zonnepalen_map():
    m = folium.Map(
        location=[52.395714, 4.792993],
        zoom_start=16,
        zoom_control=False,
        scrollWheelZoom=False,
        dragging=False
    )

    for i in df_2.index:
        LAT = df_2.loc[i, 'LAT']
        LNG = df_2.loc[i, 'LNG']
        size = df_2.loc[i, 'nl2023_panelen.1']

        folium.CircleMarker(
            location=(LAT, LNG),
            radius=0.01 * size,
            fill_color='red',
            tooltip='<b>Klik hier om de popup te zien</b>',
            popup=f"Aantal zonnepanelen = {size}"
        ).add_to(m)

    return m


map_display = aantal_zonnepalen_map()
st_folium(map_display, width=1600)


st.title('Vermogen van zonnepanelen per bedrijf')
st.write(
    '''
    De hoeveelheid zonnepanelen vertellen niet het hele verhaal. 
    Sommige zonnepanelen zijn beter dan de andere.
    De gemeente Amsterdam bevat ook data over de hoeveelheid vermogen die wordt gegenereerd.
    '''
)

def heatmap_map():
    m = folium.Map(
        location=[52.395714, 4.792993],
        zoom_start=16,
        zoom_control=False,
        scrollWheelZoom=False,
        dragging=False
    )

    HeatMap(df_2[['LAT', 'LNG', 'nl2023_wp']].values.tolist()).add_to(m)

    return m


map_display_2 = heatmap_map()
st_folium(map_display_2, width=1600)



df_zonnepanelen = eigen_excel_file('Aansluiting netbeheer')


st.title('Oppervlakte en energie van zonnepanelen per bedrijf in Sloterdijk Poort Noord')
st.write(
    '''
    Met behulp van Google Earth, zijn de oppervlakten van de daken en zonnepanelen berekend.
    Dit is per bedrijf handmatig gemeten. 
    In het geval dat meerdere bedrijven in hetzelfde gebouw bevinden, wordt één bedrijf benoemt.
    Daarbij kan de opgewekte energie worden berekend.
    Maximale opwekbare energie is gebaseerd op een gebruik van 85% bezettingsgraad van zonnepanelen.
    '''
)

plot_choice = st.radio(
    "Selecteer de weergave:",
    ("Oppervlakte [m^2]", "Opgewekte energie [kWh]")
)

fig = go.Figure()

if plot_choice == "Oppervlakte [m^2]":

    fig.add_trace(go.Bar(
        x=df_zonnepanelen['Bedrijf'],
        y=df_zonnepanelen['Zonnepaneel Oppervlakte [m^2]'],
        name='Oppervlakte zonnepanelen'
    ))

    fig.add_trace(go.Bar(
        x=df_zonnepanelen['Bedrijf'],
        y=df_zonnepanelen['Totaal Oppervlakte [m^2]'],
        name='Totaal oppervlakte'
    ))


    fig.update_layout(
        title='Oppervlakte van zonnepanelen en totaal per bedrijf',
        xaxis_title='Bedrijf',
        yaxis_title='Oppervlakte [m^2]',
        barmode='group'
    )

elif plot_choice == "Opgewekte energie [kWh]":
    fig.add_trace(go.Bar(
        x=df_zonnepanelen['Bedrijf'],
        y=df_zonnepanelen['Vermogen Zonnepaneel Oppervlakte [kWh]'],
        name='Huidige opwekbare energie'
    ))

    fig.add_trace(go.Bar(
        x=df_zonnepanelen['Bedrijf'],
        y=df_zonnepanelen['Vermogen 85% Oppervlakte [kWh]'],
        name='Maximale opwekbare energie'
    ))

    fig.update_layout(
        title='Opgewekte energie van zonnepanelen per bedrijf',
        xaxis_title='Bedrijf',
        yaxis_title='Energie [kWh]',
        barmode='group'
    )

st.plotly_chart(fig)

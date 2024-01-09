import warnings

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from statsmodels.tsa.arima.model import ARIMA

# Ignorowanie wszystkich ostrzeżeń użytkownika
warnings.filterwarnings('ignore', category=UserWarning)
matplotlib.use('TkAgg')  # Zmiana backendu na 'TkAgg'


def algorithms_data_to_csv():
    models = [('Decision Tree', dt_model, dt_y_pred),
              ('Logistic Regression', lr_model, lr_y_pred),
              ('SVC', svc_model, svc_y_pred)]
    results_df = pd.DataFrame(columns=['Model', 'Borough', 'Predicted Complaint'])
    for model_name, model, pred in models[:3]:
        if pred is not None:
            results = classifier.generate_results_dataframe(pred)
            if 'Borough' in results and results['Borough'].dtype == 'object':
                results['Model'] = model_name
                results_df = pd.concat([results_df, results], ignore_index=True)
            else:
                print(f"Skipping '{model_name}' - 'Borough' inverse transform due to inconsistent data type.")
    if 'Borough' in results_df:
        try:
            results_df['Borough'] = classifier.le_borough.inverse_transform(results_df['Borough'])
        except Exception as e:
            print(f"Error: {e}. Unable to perform 'Borough' inverse transform.")
    results_df = results_df.drop_duplicates().groupby('Predicted Complaint').head(10)
    results_df.to_csv('results_from_model.csv', index=False)


def save_complaints_per_district_yearly_to_html(complaints):
    # Tworzenie wykresu słupkowego za pomocą biblioteki Plotly Express
    fig = px.bar(complaints,
                 x='Year',
                 y='Rating',
                 color='Borough',
                 labels={'Year': 'Rok', 'Rating': 'Ocena', 'Borough': 'Dzielnica'},
                 title='Ocena Skarg na Dzielnice w Poszczególnych Latach')

    # Konfiguracja układu wykresu
    fig.update_layout(barmode='group')

    # Zapis wykresu do pliku HTML
    fig.write_html('complaints_per_district_yearly.html')


class ComplaintsClassifier:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = pd.read_csv(filepath, low_memory=False)
        self.le_borough = LabelEncoder()
        self.le_complaint = LabelEncoder()
        self.le = None
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.missing_percentage_rounded = None

    def preprocess_data(self):
        # Wczytanie zestawu danych
        file_path = "Service_Requests_311.csv"
        df = pd.read_csv(file_path, low_memory=False)

        # Zliczanie brakujących danych w każdej kolumnie
        missing_data = df.isna().sum()
        total_rows = len(df)
        missing_percentage = (missing_data / total_rows) * 100
        self.missing_percentage_rounded = missing_percentage.round(1)
        missing_df = pd.DataFrame(
            {'Liczba pustych wartości w odpowiednej kolumnie': missing_data, '%': self.missing_percentage_rounded})
        print(missing_df)

    def generate_plot_missing_data(self):
        # Wykres słupkowy brakujących danych
        plt.figure(figsize=(15, 8))
        self.missing_percentage_rounded.plot(kind='bar')
        plt.title('Procent brakujących danych w każdej kolumnie')
        plt.ylabel('Procent brakujących danych')
        plt.xticks(rotation=90)
        plt.savefig('missing_data_bar_chart.png')  # Zapisanie wykresu jako pliku

    def prepare_data_for_algorithms(self):
        # Obsługa brakujących danych
        df_filled = self.df.fillna(0)  # Wypełnienie brakujących danych wartością domyślną

        # Konwersja kolumn kategorycznych na numeryczne za pomocą LabelEncoder
        # LabelEncoder zamienia wartości tekstowe na liczby, co jest niezbędne dla modeli ML
        self.le = LabelEncoder()
        for column in df_filled.columns:
            if df_filled[column].dtype == 'object':
                df_filled[column] = self.le.fit_transform(df_filled[column].astype(str))

        # Ograniczenie zbioru danych do losowego podzbioru (np. 1% oryginalnych danych)
        df_sample = df_filled.sample(frac=0.01, random_state=42)

        # Podział ograniczonego zestawu danych na część do treningu i testowania modelu
        x = df_sample.drop('Complaint Type', axis=1)  # Cechy (bez kolumny 'Complaint Type')
        y = df_sample['Complaint Type']  # Zmienna docelowa (tylko 'Complaint Type')
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        # Skalowanie danych przed trenowaniem modeli - potrzebne dla Modelu Regresji Logistycznej
        # ze względu na to że algorytm nie zbiega się w ustalonej liczbie iteracji.
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)

        # Zakoduj etykiety za pomocą LabelEncoder
        self.le_borough = LabelEncoder()
        self.df['Borough'] = self.le_borough.fit_transform(self.df['Borough'])

        self.le_complaint = LabelEncoder()
        self.df['Complaint Type'] = self.le_complaint.fit_transform(self.df['Complaint Type'])

    # Użycie trzech różnych algorytmów do porównania
    # Trenowanie i predykcja za pomocą Drzewa Decyzyjnego
    def decision_tree(self):
        model = DecisionTreeClassifier()
        model.fit(self.X_train, self.y_train)
        pred = model.predict(self.X_test)
        dt_accuracy = accuracy_score(self.y_test, pred)
        print(f"\nDokładność modelu drzewa decyzyjnego: {round(dt_accuracy * 100, 1)}")
        self.generate_plot_results_city(pred, 'decision_tree')
        return model, pred

    # Trenowanie i predykcja za pomocą Regresji Logistycznej
    def logistic_regression(self):
        model = LogisticRegression(max_iter=10000)
        model.fit(self.X_train_scaled, self.y_train)
        pred = model.predict(self.X_test_scaled)
        lr_accuracy = accuracy_score(self.y_test, pred)
        print(f"Dokładność modelu regresji logistycznej: {round(lr_accuracy * 100, 1)}")
        self.generate_plot_results_city(pred, 'logistic_regression')
        return model, pred

    # Trenowanie i predykcja za pomocą SVC
    def svc(self):
        model = SVC(kernel='linear')
        model.fit(self.X_train_scaled, self.y_train)
        y_pred = model.predict(self.X_test_scaled)
        svc_accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Dokładność modelu SVC: {round(svc_accuracy * 100, 1)}")
        self.generate_plot_results_city(y_pred, 'svc')
        return model, y_pred

    def generate_plot_results_city(self, y_pred, model):
        # Przeprowadzenie predykcji na całym zestawie danych
        all_predictions = y_pred

        # Przekształcenie przewidzianych etykiet na czytelne nazwy
        all_pred_labels = self.le_complaint.inverse_transform(all_predictions)

        # Zliczanie wystąpień każdego przewidywanego typu skargi
        predicted_counts = pd.Series(all_pred_labels).value_counts()

        # Wybór 5 najczęściej przewidywanych typów skarg
        top_5_predicted = predicted_counts.nlargest(5)

        # Konwersja na DataFrame i ustawienie nazw problemów jako indeks
        top_5_predicted_df = top_5_predicted.to_frame(name='Number of Predictions').reset_index()
        top_5_predicted_df.columns = ['Complaint Type', 'Number of Predictions']

        # Tworzenie wykresu słupkowego
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Complaint Type', y='Number of Predictions', data=top_5_predicted_df)
        plt.title(f'Top 5 problemów dla miasta - ' + model)
        plt.xlabel('Complaint Type')
        plt.ylabel('Number of Predictions')
        plt.xticks(rotation=45, ha='right')  # Obrócenie etykiet dla lepszej czytelności
        plt.tight_layout()
        plt.savefig(model + '_city_results.png')

    def generate_results_dataframe(self, y_pred):
        """
        Tworzy DataFrame z wynikami predykcji dla każdej z dzielnic.
        """
        y_pred_labels = self.le_complaint.inverse_transform(y_pred)
        x_test_borough_labels = self.le_borough.inverse_transform(self.X_test['Borough'])

        results_df = pd.DataFrame({'Borough': x_test_borough_labels, 'Predicted Complaint': y_pred_labels})
        return results_df

    def generate_advanced_plot_results(self, y_pred, model_type):
        # Generowanie ramki danych z wynikami modelu
        results_df = self.generate_results_dataframe(y_pred)

        # Obliczanie procentowej liczby zgłoszeń dla każdej kombinacji Borough i Predicted Complaint
        results_count = results_df.groupby(['Borough', 'Predicted Complaint']).size().reset_index(name='Counts')
        borough_total = results_count.groupby('Borough')['Counts'].transform('sum')
        results_count['Percentage'] = (results_count['Counts'] / borough_total) * 100

        # Tworzenie wykresu słupkowego z użyciem biblioteki Plotly Express
        fig = px.bar(results_count, x='Borough', y='Percentage', color='Predicted Complaint', barmode='group',
                     labels={'Percentage': 'Procent zgłoszeń'})
        fig.update_layout(title='Rozkład przewidywanych problemów w dzielnicach', xaxis_title='Dzielnica',
                          yaxis_title='Procent zgłoszeń')
        fig.write_html('advanced_plot_results_' + model_type + '.html')  # Zapisanie wykresu jako interaktywny plik HTML

    # Model ARIMA (Autoregressive Integrated Moving Average) jest modelem statystycznym
    # wykorzystywanym do analizy i prognozowania szeregów czasowych.
    # W moim kodzie, model ARIMA jest dopasowywany do szeregu czasowego wygenerowanego
    # z przewidywanych wartości zgłoszeń.
    def time_series_forecast(self, y_pred, model_type):
        # Przygotowanie DataFrame z przewidywaniami i ich datami
        pred_df = pd.DataFrame({'Predicted': y_pred}, index=self.X_test.index)

        # Połączenie z oryginalnym DataFrame
        merged_df = pd.concat([self.df, pred_df], axis=1)

        # Konwersja indeksu na DatetimeIndex
        merged_df['Created Date'] = pd.to_datetime(merged_df['Created Date'], errors='coerce')
        merged_df.dropna(subset=['Created Date'], inplace=True)
        merged_df.set_index('Created Date', inplace=True)

        # Agregowanie przewidzianych wartości na każdy dzień
        daily_predicted_counts = merged_df['Predicted'].resample('D').sum()

        # Prognoza ARIMA
        model = ARIMA(daily_predicted_counts.dropna(), order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=30)
        forecast_dates = pd.date_range(start=daily_predicted_counts.dropna().index[-1], periods=31, freq='D')[1:]

        # Wykres prognozy ARIMA
        forecast_fig = go.Figure()
        forecast_fig.add_trace(
            go.Scatter(x=daily_predicted_counts.dropna().index[-60:], y=daily_predicted_counts.dropna()[-60:],
                       mode='lines', name='Original'))
        forecast_fig.add_trace(
            go.Scatter(x=forecast_dates, y=forecast, mode='lines', name='Forecast', line=dict(color='red')))

        forecast_fig.update_layout(title=f'ARIMA Forecast - {model_type}', xaxis_title='Date',
                                   yaxis_title='Predicted Counts')
        forecast_fig.update_xaxes(tickangle=-45, nticks=20)

        # Zapis wykresu prognozy do pliku HTML
        forecast_fig.write_html(f'arima_forecast_{model_type}.html')

    def complaints_per_district_yearly(self, include_complaint_type=True, include_borough=True):
        # Konwertowanie kolumny 'Created Date' na typ daty
        self.df['Created Date'] = pd.to_datetime(self.df['Created Date'])

        # Tworzenie kolumny 'Year' na podstawie roku z kolumny 'Created Date'
        self.df['Year'] = self.df['Created Date'].dt.year

        # Wybór kolumn do uwzględnienia w grupowaniu
        columns_to_group_by = ['Year']
        if include_borough:
            columns_to_group_by.append('Borough')
        if include_complaint_type:
            columns_to_group_by.append('Complaint Type')

        # Grupowanie po odpowiednich kolumnach dla uzyskania liczby skarg
        complaints_by_year_and_district = self.df.groupby(columns_to_group_by).size().reset_index(name='Complaints')

        # Obliczenie łącznej liczby skarg dla każdego roku w celu wyznaczenia oceny
        total_complaints_per_year = complaints_by_year_and_district.groupby('Year')['Complaints'].sum()

        # Wyznaczenie oceny dla każdej dzielnicy w każdym roku
        complaints_by_year_and_district['Rating'] = complaints_by_year_and_district.apply(
            lambda x: x['Complaints'] / total_complaints_per_year[x['Year']] * 100, axis=1
        )

        return complaints_by_year_and_district

    def save_all_to_csv(self, complaints):
        # Pobranie oryginalnych nazw etykiet dla kolumn Complaint Type i Borough
        complaint_type_labels = self.le_complaint.inverse_transform(complaints['Complaint Type'])
        borough_labels = self.le_borough.inverse_transform(complaints['Borough'])

        # Utworzenie kopii DataFrame z prawidłowymi nazwami etykiet
        complaints_with_labels = complaints.copy()
        complaints_with_labels['Complaint Type'] = complaint_type_labels
        complaints_with_labels['Borough'] = borough_labels

        # Grupowanie i wybieranie top 5 problemów dla każdej dzielnicy
        top_5_per_district = complaints_with_labels.groupby('Borough').apply(
            lambda x: x.nlargest(5, 'Complaints')).reset_index(drop=True)

        # Zapisanie ocen skarg do pliku CSV
        top_5_per_district.to_csv('top_5_complaints_per_district.csv', index=False)


if __name__ == "__main__":
    # Tworzymy instancję ComplaintsClassifier
    classifier = ComplaintsClassifier("Service_Requests_311.csv")

    # Korzystamy z metod
    classifier.preprocess_data()
    classifier.generate_plot_missing_data()
    classifier.prepare_data_for_algorithms()

    # Przeliczenie ratingu skarg dla każdej dzielnicy z rozbiciem na lata
    complaints_per_district_yearly = classifier.complaints_per_district_yearly()
    # Zapisanie wyników do pliku HTML
    save_complaints_per_district_yearly_to_html(complaints_per_district_yearly)

    dt_model, dt_y_pred = classifier.decision_tree()
    classifier.generate_advanced_plot_results(dt_y_pred, 'decision_tree')
    classifier.time_series_forecast(dt_y_pred, 'decision_tree')

    lr_model, lr_y_pred = classifier.logistic_regression()
    classifier.generate_advanced_plot_results(lr_y_pred, 'logistic_regression')
    classifier.time_series_forecast(lr_y_pred, 'logistic_regression')

    svc_model, svc_y_pred = classifier.svc()
    classifier.generate_advanced_plot_results(svc_y_pred, 'svc')
    classifier.time_series_forecast(svc_y_pred, 'svc')

    classifier.save_all_to_csv(classifier.complaints_per_district_yearly())

    algorithms_data_to_csv()

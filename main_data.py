import warnings

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

# Ignorowanie wszystkich ostrzeżeń użytkownika
warnings.filterwarnings('ignore', category=UserWarning)
matplotlib.use('TkAgg')  # Zmiana backendu na 'TkAgg'


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
        df_sample = df_filled.sample(frac=0.0001, random_state=42)

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
        dt_model = DecisionTreeClassifier()
        dt_model.fit(self.X_train, self.y_train)
        dt_y_pred = dt_model.predict(self.X_test)
        dt_accuracy = accuracy_score(self.y_test, dt_y_pred)
        print(f"\nDokładność modelu drzewa decyzyjnego: {round(dt_accuracy * 100, 1)}")
        self.generate_plot_results_city(dt_y_pred, 'decision_tree')
        return dt_model, dt_y_pred

    # Trenowanie i predykcja za pomocą Regresji Logistycznej
    def logistic_regression(self):
        lr_model = LogisticRegression(max_iter=10000)
        lr_model.fit(self.X_train_scaled, self.y_train)
        lr_y_pred = lr_model.predict(self.X_test_scaled)
        lr_accuracy = accuracy_score(self.y_test, lr_y_pred)
        print(f"Dokładność modelu regresji logistycznej: {round(lr_accuracy * 100, 1)}")
        self.generate_plot_results_city(lr_y_pred, 'logistic_regression')
        return lr_model, lr_y_pred

    # Trenowanie i predykcja za pomocą SVM
    def svc(self):
        svm_model = SVC(kernel='linear')
        svm_model.fit(self.X_train_scaled, self.y_train)
        svm_y_pred = svm_model.predict(self.X_test_scaled)
        svm_accuracy = accuracy_score(self.y_test, svm_y_pred)
        print(f"Dokładność modelu SVM: {round(svm_accuracy * 100, 1)}")
        self.generate_plot_results_city(svm_y_pred, 'svc')
        return svm_model, svm_y_pred

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

    def generate_results_dataframe(self, model, y_pred):
        """
        Tworzy DataFrame z wynikami predykcji dla każdej z dzielnic.
        """
        y_pred_labels = self.le_complaint.inverse_transform(y_pred)
        x_test_borough_labels = self.le_borough.inverse_transform(self.X_test['Borough'])

        results_df = pd.DataFrame({'Borough': x_test_borough_labels, 'Predicted Complaint': y_pred_labels})
        return results_df

    def generate_advanced_plot_results(self, model, y_pred, type):
        # Generowanie ramki danych z wynikami modelu
        results_df = self.generate_results_dataframe(model, y_pred)

        # Obliczanie procentowej liczby zgłoszeń dla każdej kombinacji Borough i Predicted Complaint
        results_count = results_df.groupby(['Borough', 'Predicted Complaint']).size().reset_index(name='Counts')
        borough_total = results_count.groupby('Borough')['Counts'].transform('sum')
        results_count['Percentage'] = (results_count['Counts'] / borough_total) * 100

        # Tworzenie wykresu słupkowego z użyciem biblioteki Plotly Express
        fig = px.bar(results_count, x='Borough', y='Percentage', color='Predicted Complaint', barmode='group',
                     labels={'Percentage': 'Procent zgłoszeń'})
        fig.update_layout(title='Rozkład przewidywanych problemów w dzielnicach', xaxis_title='Dzielnica',
                          yaxis_title='Procent zgłoszeń')
        fig.write_html('advanced_plot_results_' + type + '.html')  # Zapisanie wykresu jako interaktywny plik HTML

    # Model ARIMA (Autoregressive Integrated Moving Average) jest modelem statystycznym
    # wykorzystywanym do analizy i prognozowania szeregów czasowych.
    # W moim kodzie, model ARIMA jest dopasowywany do szeregu czasowego wygenerowanego
    # z przewidywanych wartości zgłoszeń.
    def time_series_decomposition_and_forecast(self, y_pred, model_type):
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

        # Dekompozycja serii czasowej
        decomposition = seasonal_decompose(daily_predicted_counts.dropna(), model='additive')

        # Tworzenie subwykresów dla dekompozycji
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                            subplot_titles=(
                                f'Trend - {model_type}', f'Sezonowość - {model_type}', f'Reszta - {model_type}'))

        # Wykres składowej Trend
        fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, mode='lines', name='Trend'), row=1,
                      col=1)

        # Wykres składowej Seasonal
        fig.add_trace(
            go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, mode='lines', name='Seasonal'), row=2,
            col=1)

        # Wykres składowej Residual
        fig.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, mode='lines', name='Residual'),
                      row=3, col=1)

        # Aktualizacja układu wykresu
        fig.update_layout(height=800, title_text=f"Time Series Decomposition - {model_type}")
        fig.update_xaxes(tickangle=-45, nticks=20)

        # Zapis do pliku HTML
        fig.write_html(f'decomposition_{model_type}.html')

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

    def save_complaints_per_district_yearly_to_html(self, complaints_per_district_yearly):
        # Tworzenie wykresu słupkowego za pomocą biblioteki Plotly Express
        fig = px.bar(complaints_per_district_yearly,
                     x='Year',
                     y='Rating',
                     color='Borough',
                     labels={'Year': 'Rok', 'Rating': 'Ocena', 'Borough': 'Dzielnica'},
                     title='Ocena Skarg na Dzielnice w Poszczególnych Latach')

        # Konfiguracja układu wykresu
        fig.update_layout(barmode='group')

        # Zapis wykresu do pliku HTML
        fig.write_html('complaints_per_district_yearly.html')

    def save_all_to_csv(self, complaints_per_district_yearly):
        # Pobranie oryginalnych nazw etykiet dla kolumn Complaint Type i Borough
        complaint_type_labels = self.le_complaint.inverse_transform(complaints_per_district_yearly['Complaint Type'])
        borough_labels = self.le_borough.inverse_transform(complaints_per_district_yearly['Borough'])

        # Utworzenie kopii DataFrame z prawidłowymi nazwami etykiet
        complaints_with_labels = complaints_per_district_yearly.copy()
        complaints_with_labels['Complaint Type'] = complaint_type_labels
        complaints_with_labels['Borough'] = borough_labels

        # Zapisanie ocen skarg do pliku CSV
        complaints_with_labels.to_csv('complaints_per_district_yearly.csv', index=False)


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
    classifier.save_complaints_per_district_yearly_to_html(complaints_per_district_yearly)

    dt_model, dt_y_pred = classifier.decision_tree()
    classifier.generate_advanced_plot_results(dt_model, dt_y_pred, 'decision_tree')
    classifier.time_series_decomposition_and_forecast(dt_y_pred, 'decision_tree')

    lr_model, lr_y_pred = classifier.logistic_regression()
    classifier.generate_advanced_plot_results(lr_model, lr_y_pred, 'logistic_regression')
    classifier.time_series_decomposition_and_forecast(lr_y_pred, 'logistic_regression')

    svm_model, svm_y_pred = classifier.svc()
    classifier.generate_advanced_plot_results(svm_model, svm_y_pred, 'svc')
    classifier.time_series_decomposition_and_forecast(svm_y_pred, 'svc')

    classifier.save_all_to_csv(classifier.complaints_per_district_yearly())

    # Saving data using models and pred variables to CSV
    models = [('Decision Tree', dt_model, dt_y_pred),
              ('Logistic Regression', lr_model, lr_y_pred),
              ('SVM', svm_model, svm_y_pred)]

    results_df = pd.DataFrame(columns=['Model', 'Borough', 'Predicted Complaint'])

    for model_name, model, pred in models[:3]:
        if pred is not None:
            results = classifier.generate_results_dataframe(model, pred)
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
    results_df.to_csv('results_from_models_with_model.csv', index=False)

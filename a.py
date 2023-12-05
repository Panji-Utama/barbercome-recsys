from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

stop_factory = StopWordRemoverFactory()
more_stopword = ['dengan', 'ia','bahwa','oleh']
fix_stopword = stop_factory.get_stop_words() + more_stopword
# indonesian_stop_words = ['dan', 'yang', 'di', 'dari', 'untuk', 'pada', 'ke', 'karena', ...]

def preprocess_indonesian(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    words = text.split()
    words = [word for word in words if word not in fix_stopword]
    return ' '.join(words)

user_preferences = [
    # "Saya suka potongan rambut modern seperti fade dan undercuts",
    # "Saya lebih memilih gaya rambut klasik dan cukuran yang rapi",
    # "Ingin mencoba gaya rambut panjang ala Korea",
    # "Saya suka potongan yang sederhana, cepat dan bersih",
    # "Saya mencari barber yang bisa melakukan grooming jenggot",
    "Saya mencari barber yang dapat memberikan potongan fade, dan juga membuat jenggot saya tipis dan bersih",
    "Saya mau selalu botak dan bersih",
]

barber_services = [
    "Kami menawarkan potongan rambut modern seperti fade, pompadour, dan undercuts",
    "Spesialis gaya rambut klasik dan cukuran tradisional",
    "Pakar dalam gaya rambut panjang dan styling ala Korea",
    "Melayani potongan rambut cepat dan bersih untuk gaya sederhana",
    "Menyediakan jasa grooming jenggot dan perawatan wajah",
    "Kami memiliki paket lengkap untuk grooming dan potongan rambut trendi",
    "Barbershop kami adalah pilihan tepat untuk potongan rambut pria dan anak-anak",
    "Kami ahli dalam berbagai gaya rambut, dari klasik hingga modern",
]


user_preferences = [preprocess_indonesian(text) for text in user_preferences]
barber_services = [preprocess_indonesian(text) for text in barber_services]

# Combine and vectorize the data
all_text = user_preferences + barber_services
print(all_text)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(all_text)


# Calculate cosine similarity
# Assuming first N entries in tfidf_matrix are users, and the rest are barbers
N = len(user_preferences)
user_vectors = tfidf_matrix[:N]
barber_vectors = tfidf_matrix[N:]

# Compute similarity between each user preference and barber service
similarity_scores = cosine_similarity(user_vectors, barber_vectors)


# Example: Print top 3 recommendations for each user
top_n = 3
for user_index in range(len(user_preferences)):
    user_similarity_scores = similarity_scores[user_index]
    
    # Get indices of top matching services
    top_service_indices = user_similarity_scores.argsort()[-top_n:][::-1]

    print(f"Recommendations for User {user_index + 1} ({user_preferences[user_index]}):")
    for i in top_service_indices:
        corresponding_service = barber_services[i]
        print(f"   Barber Service {i + 1}: {corresponding_service} (Score: {user_similarity_scores[i]:.2f})")
    print("\n")

from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

full_indonesian_stopword = [
    "ada", "adalah", "adanya", "adapun", "agak", "agaknya", "agar", "akan", "akankah", "akhir",
    "akhiri", "akhirnya", "aku", "akulah", "amat", "amatlah", "anda", "andalah", "antar", "antara",
    "antaranya", "apa", "apaan", "apabila", "apakah", "apalagi", "apatah", "artinya", "asal",
    "asalkan", "atas", "atau", "ataukah", "ataupun", "awal", "awalnya", "bagai", "bagaikan",
    "bagaimana", "bagaimanakah", "bagaimanapun", "bagi", "bagian", "bahkan", "bahwa", "bahwasanya",
    "baik", "bakal", "bakalan", "balik", "banyak", "bapak", "baru", "bawah", "beberapa", "begini",
    "beginian", "beginikah", "beginilah", "begitu", "begitukah", "begitulah", "begitupun", "bekerja",
    "belakang", "belakangan", "belum", "belumlah", "benar", "benarkah", "benarlah", "berada", "berakhir",
    "berakhirlah", "berakhirnya", "berapa", "berapakah", "berapalah", "berapapun", "berarti", "berawal",
    "berbagai", "berdatangan", "beri", "berikan", "berikut", "berikutnya", "berjumlah", "berkali-kali",
    "berkata", "berkehendak", "berkeinginan", "berkenaan", "berlainan", "berlalu", "berlangsung",
    "berlebihan", "bermacam", "bermacam-macam", "bermaksud", "bermula", "bersama", "bersama-sama",
    "bersiap", "bersiap-siap", "bertanya", "bertanya-tanya", "berturut", "berturut-turut", "bertutur",
    "berujar", "berupa", "besar", "betul", "betulkah", "biasa", "biasanya", "bila", "bilakah", "bisa",
    "bisakah", "boleh", "bolehkah", "bolehlah", "buat", "bukan", "bukankah", "bukanlah", "bukannya",
    "bulan", "bung", "cara", "caranya", "cukup", "cukupkah", "cukuplah", "cuma", "dahulu", "dalam",
    "dan", "dapat", "dari", "daripada", "datang", "dekat", "demi", "demikian", "demikianlah", "dengan",
    "depan", "di", "dia", "diakhiri", "diakhirinya", "dialah", "diantara", "diantaranya", "diberi",
    "diberikan", "diberikannya", "dibuat", "dibuatnya", "didapat", "didatangkan", "digunakan",
    "diibaratkan", "diibaratkannya", "diingat", "diingatkan", "diinginkan", "dijawab", "dijelaskan",
    "dijelaskannya", "dikarenakan", "dikatakan", "dikatakannya", "dikerjakan", "diketahui",
    "diketahuinya", "dikira", "dilakukan", "dilalui", "dilihat", "dimaksud", "dimaksudkan",
    "dimaksudkannya", "dimaksudnya", "diminta", "dimintai", "dimisalkan", "dimulai", "dimulailah",
    "dimulainya", "dimungkinkan", "dini", "dipastikan", "diperbuat", "diperbuatnya", "dipergunakan",
    "diperkirakan", "diperlihatkan", "diperlukan", "diperlukannya", "dipersoalkan", "dipertanyakan",
    "dipunyai", "diri", "dirinya", "disampaikan", "disebut", "disebutkan", "disebutkannya", "disini",
    "disinilah", "ditambahkan", "ditandaskan", "ditanya", "ditanyai", "ditanyakan", "ditegaskan",
    "ditujukan", "ditunjuk", "ditunjuki", "ditunjukkan", "ditunjukkannya", "ditunjuknya", "dituturkan",
    "dituturkannya", "diucapkan", "diucapkannya", "diungkapkan", "dong", "dua", "dulu", "empat", "enggak",
    "enggaknya", "entah", "entahlah", "guna", "gunakan", "hal", "hampir", "hanya", "hanyalah", "hari",
    "harus", "haruslah", "harusnya", "hendak", "hendaklah", "hendaknya", "hingga", "ia", "ialah", "ibarat",
    "ibaratkan", "ibaratnya", "ibu", "ikut", "ingat", "ingat-ingat", "ingin", "inginkah", "inginkan",
    "ini", "inikah", "inilah", "itu", "itukah", "itulah", "jadi", "jadilah", "jadinya", "jangan", "jangankan",
    "janganlah", "jauh", "jawab", "jawaban", "jawabnya", "jelas", "jelaskan", "jelaslah", "jelasnya", "jika",
    "jikalau", "juga", "jumlah", "jumlahnya", "justru", "kala", "kalau", "kalaulah", "kalaupun", "kalian",
    "kami", "kamilah", "kamu", "kamulah", "kan", "kapan", "kapankah", "kapanpun", "karena", "karenanya",
    "kasus", "kata", "katakan", "katakanlah", "katanya", "ke", "keadaan", "kebetulan", "kecil", "kedua",
    "keduanya", "keinginan", "kelamaan", "kelihatan", "kelihatannya", "kelima", "keluar", "kembali",
    "kemudian", "kemungkinan", "kemungkinannya", "kenapa", "kepada", "kepadanya", "kesampaian", "keseluruhan",
    "keseluruhannya", "keterlaluan", "ketika", "khususnya", "kini", "kinilah", "kira", "kira-kira", "kiranya",
    "kita", "kitalah", "kok", "kurang", "lagi", "lagian", "lah", "lain", "lainnya", "lalu", "lama", "lamanya",
    "lanjut", "lanjutnya", "lebih", "lewat", "lima", "luar", "macam", "maka", "makanya", "makin", "malah",
    "malahan", "mampu", "mampukah", "mana", "manakala", "manalagi", "masa", "masalah", "masalahnya", "masih",
    "masihkah", "masing", "masing-masing", "mau", "maupun", "melainkan", "melakukan", "melalui", "melihat",
    "melihatnya", "memang", "memastikan", "memberi", "memberikan", "membuat", "memerlukan", "memihak",
    "meminta", "memintakan", "memisalkan", "memperbuat", "mempergunakan", "memperkirakan", "memperlihatkan",
    "mempersiapkan", "mempersoalkan", "mempertanyakan", "mempunyai", "memulai", "memungkinkan", "menaiki",
    "menambahkan", "menandaskan", "menanti", "menanti-nanti", "menantikan", "menanya", "menanyai",
    "menanyakan", "mendapat", "mendapatkan", "mendatang", "mendatangi", "mendatangkan", "menegaskan",
    "mengakhiri", "mengapa", "mengatakan", "mengatakannya", "mengenai", "mengerjakan", "mengetahui",
    "menggunakan", "menghendaki", "mengibaratkan", "mengibaratkannya", "mengingat", "mengingatkan",
    "menginginkan", "mengira", "mengucapkan", "mengucapkannya", "mengungkapkan", "menjadi", "menjawab",
    "menjelaskan", "menuju", "menunjuk", "menunjuki", "menunjukkan", "menunjuknya", "menurut", "menuturkan",
    "menyampaikan", "menyangkut", "menyatakan", "menyebutkan", "menyeluruh", "menyiapkan", "merasa",
    "mereka", "merekalah", "merupakan", "meski", "meskipun", "meyakini", "meyakinkan", "minta", "mirip",
    "misal", "misalkan", "misalnya", "mula", "mulai", "mulailah", "mulanya", "mungkin", "mungkinkah", "nah",
    "naik", "namun", "nanti", "nantinya", "nyaris", "nyatanya", "oleh", "olehnya", "pada", "padahal",
    "padanya", "pak", "paling", "panjang", "pantas", "para", "pasti", "pastilah", "penting", "pentingnya",
    "per", "percuma", "perlu", "perlukah", "perlunya", "pernah", "persoalan", "pertama", "pertama-tama",
    "pertanyaan", "pertanyakan", "pihak", "pihaknya", "pukul", "pula", "pun", "punya", "rasa", "rasanya",
    "rata", "rupanya", "saat", "saatnya", "saja", "sajalah", "saling", "sama", "sama-sama", "sambil", "sampai",
    "sampai-sampai", "sampaikan", "sana", "sangat", "sangatlah", "satu", "saya", "sayalah", "se", "sebab",
    "sebabnya", "sebagai", "sebagaimana", "sebagainya", "sebagian", "sebaik", "sebaik-baiknya", "sebaiknya",
    "sebaliknya", "sebanyak", "sebegini", "sebegitu", "sebelum", 'sebelumnya', 'sebenarnya', 'seberapa', 'sebesar', 'sebetulnya', 'sebisanya', 'sebuah', 'sebut', 'sebutlah', 'sebutnya',
    'secara', 'secukupnya', 'sedang', 'sedangkan', 'sedemikian', 'sedikit', 'sedikitnya', 'seenaknya', 'segala', 'segalanya',
    'segera', 'seharusnya', 'sehingga', 'seingat', 'sejak', 'sejauh', 'sejenak', 'sejumlah', 'sekadar', 'sekadarnya', 'sekali',
    'sekali-kali', 'sekalian', 'sekaligus', 'sekalipun', 'sekarang', 'sekarang', 'sekecil', 'seketika', 'sekiranya', 'sekitar',
    'sekitarnya', 'sekurang-kurangnya', 'sekurangnya', 'sela', 'selain', 'selaku', 'selalu', 'selama', 'selama-lamanya',
    'selamanya', 'selanjutnya', 'seluruh', 'seluruhnya', 'semacam', 'semakin', 'semampu', 'semampunya', 'semasa', 'semasih',
    'semata', 'semata-mata', 'semaunya', 'sementara', 'semisal', 'semisalnya', 'sempat', 'semua', 'semuanya', 'semula', 'sendiri',
    'sendirian', 'sendirinya', 'seolah', 'seolah-olah', 'seorang', 'sepanjang', 'sepantasnya', 'sepantasnyalah', 'seperlunya',
    'seperti', 'sepertinya', 'sepihak', 'sering', 'seringnya', 'serta', 'serupa', 'sesaat', 'sesama', 'sesampai', 'sesegera',
    'sesekali', 'seseorang', 'sesuatu', 'sesuatunya', 'sesudah', 'sesudahnya', 'setelah', 'setempat', 'setengah', 'seterusnya',
    'setiap', 'setiba', 'setibanya', 'setidak-tidaknya', 'setidaknya', 'setinggi', 'seusai', 'sewaktu', 'siap', 'siapa', 'siapakah',
    'siapapun', 'sini', 'sinilah', 'soal', 'soalnya', 'suatu', 'sudah', 'sudahkah', 'sudahlah', 'supaya', 'tadi', 'tadinya', 'tahu',
    'tahun', 'tak', 'tambah', 'tambahnya', 'tampak', 'tampaknya', 'tandas', 'tandasnya', 'tanpa', 'tanya', 'tanyakan', 'tanyanya',
    'tapi', 'tegas', 'tegasnya', 'telah', 'tempat', 'tengah', 'tentang', 'tentu', 'tentulah', 'tentunya', 'tepat', 'terakhir', 'terasa',
    'terbanyak', 'terdahulu', 'terdapat', 'terdiri', 'terhadap', 'terhadapnya', 'teringat', 'teringat-ingat', 'terjadi', 'terjadilah',
    'terjadinya', 'terkira', 'terlalu', 'terlebih', 'terlihat', 'termasuk', 'ternyata', 'tersampaikan', 'tersebut', 'tersebutlah',
    'tertentu', 'tertuju', 'terus', 'terutama', 'tetap', 'tetapi', 'tiap', 'tiba', 'tiba-tiba', 'tidak', 'tidakkah', 'tidaklah', 'tiga',
    'tinggi', 'toh', 'tunjuk', 'turut', 'tutur', 'tuturnya', 'ucap', 'ucapnya', 'ujar', 'ujarnya', 'umum', 'umumnya', 'ungkap', 'ungkapnya',
    'untuk', 'usah', 'usai', 'waduh', 'wah', 'wahai', 'waktu', 'waktunya', 'walau', 'walaupun', 'wong', 'yaitu', 'yakin', 'yakni', 'yang'
]

test = """ada
adalah
adanya
adapun
agak
agaknya
agar
akan
akankah
akhir
akhiri
akhirnya
aku
akulah
amat
amatlah
anda
andalah
antar
antara
antaranya
apa
apaan
apabila
apakah
apalagi
apatah
artinya
asal
asalkan
atas
atau
ataukah
ataupun
awal
awalnya
bagai
bagaikan
bagaimana
bagaimanakah
bagaimanapun
bagi
bagian
bahkan
bahwa
bahwasanya
baik
bakal
bakalan
balik
banyak
bapak
baru
bawah
beberapa
begini
beginian
beginikah
beginilah
begitu
begitukah
begitulah
begitupun
bekerja
belakang
belakangan
belum
belumlah
benar
benarkah
benarlah
berada
berakhir
berakhirlah
berakhirnya
berapa
berapakah
berapalah
berapapun
berarti
berawal
berbagai
berdatangan
beri
berikan
berikut
berikutnya
berjumlah
berkali-kali
berkata
berkehendak
berkeinginan
berkenaan
berlainan
berlalu
berlangsung
berlebihan
bermacam
bermacam-macam
bermaksud
bermula
bersama
bersama-sama
bersiap
bersiap-siap
bertanya
bertanya-tanya
berturut
berturut-turut
bertutur
berujar
berupa
besar
betul
betulkah
biasa
biasanya
bila
bilakah
bisa
bisakah
boleh
bolehkah
bolehlah
buat
bukan
bukankah
bukanlah
bukannya
bulan
bung
cara
caranya
cukup
cukupkah
cukuplah
cuma
dahulu
dalam
dan
dapat
dari
daripada
datang
dekat
demi
demikian
demikianlah
dengan
depan
di
dia
diakhiri
diakhirinya
dialah
diantara
diantaranya
diberi
diberikan
diberikannya
dibuat
dibuatnya
didapat
didatangkan
digunakan
diibaratkan
diibaratkannya
diingat
diingatkan
diinginkan
dijawab
dijelaskan
dijelaskannya
dikarenakan
dikatakan
dikatakannya
dikerjakan
diketahui
diketahuinya
dikira
dilakukan
dilalui
dilihat
dimaksud
dimaksudkan
dimaksudkannya
dimaksudnya
diminta
dimintai
dimisalkan
dimulai
dimulailah
dimulainya
dimungkinkan
dini
dipastikan
diperbuat
diperbuatnya
dipergunakan
diperkirakan
diperlihatkan
diperlukan
diperlukannya
dipersoalkan
dipertanyakan
dipunyai
diri
dirinya
disampaikan
disebut
disebutkan
disebutkannya
disini
disinilah
ditambahkan
ditandaskan
ditanya
ditanyai
ditanyakan
ditegaskan
ditujukan
ditunjuk
ditunjuki
ditunjukkan
ditunjukkannya
ditunjuknya
dituturkan
dituturkannya
diucapkan
diucapkannya
diungkapkan
dong
dua
dulu
empat
enggak
enggaknya
entah
entahlah
guna
gunakan
hal
hampir
hanya
hanyalah
hari
harus
haruslah
harusnya
hendak
hendaklah
hendaknya
hingga
ia
ialah
ibarat
ibaratkan
ibaratnya
ibu
ikut
ingat
ingat-ingat
ingin
inginkah
inginkan
ini
inikah
inilah
itu
itukah
itulah
jadi
jadilah
jadinya
jangan
jangankan
janganlah
jauh
jawab
jawaban
jawabnya
jelas
jelaskan
jelaslah
jelasnya
jika
jikalau
juga
jumlah
jumlahnya
justru
kala
kalau
kalaulah
kalaupun
kalian
kami
kamilah
kamu
kamulah
kan
kapan
kapankah
kapanpun
karena
karenanya
kasus
kata
katakan
katakanlah
katanya
ke
keadaan
kebetulan
kecil
kedua
keduanya
keinginan
kelamaan
kelihatan
kelihatannya
kelima
keluar
kembali
kemudian
kemungkinan
kemungkinannya
kenapa
kepada
kepadanya
kesampaian
keseluruhan
keseluruhannya
keterlaluan
ketika
khususnya
kini
kinilah
kira
kira-kira
kiranya
kita
kitalah
kok
kurang
lagi
lagian
lah
lain
lainnya
lalu
lama
lamanya
lanjut
lanjutnya
lebih
lewat
lima
luar
macam
maka
makanya
makin
malah
malahan
mampu
mampukah
mana
manakala
manalagi
masa
masalah
masalahnya
masih
masihkah
masing
masing-masing
mau
maupun
melainkan
melakukan
melalui
melihat
melihatnya
memang
memastikan
memberi
memberikan
membuat
memerlukan
memihak
meminta
memintakan
memisalkan
memperbuat
mempergunakan
memperkirakan
memperlihatkan
mempersiapkan
mempersoalkan
mempertanyakan
mempunyai
memulai
memungkinkan
menaiki
menambahkan
menandaskan
menanti
menanti-nanti
menantikan
menanya
menanyai
menanyakan
mendapat
mendapatkan
mendatang
mendatangi
mendatangkan
menegaskan
mengakhiri
mengapa
mengatakan
mengatakannya
mengenai
mengerjakan
mengetahui
menggunakan
menghendaki
mengibaratkan
mengibaratkannya
mengingat
mengingatkan
menginginkan
mengira
mengucapkan
mengucapkannya
mengungkapkan
menjadi
menjawab
menjelaskan
menuju
menunjuk
menunjuki
menunjukkan
menunjuknya
menurut
menuturkan
menyampaikan
menyangkut
menyatakan
menyebutkan
menyeluruh
menyiapkan
merasa
mereka
merekalah
merupakan
meski
meskipun
meyakini
meyakinkan
minta
mirip
misal
misalkan
misalnya
mula
mulai
mulailah
mulanya
mungkin
mungkinkah
nah
naik
namun
nanti
nantinya
nyaris
nyatanya
oleh
olehnya
pada
padahal
padanya
pak
paling
panjang
pantas
para
pasti
pastilah
penting
pentingnya
per
percuma
perlu
perlukah
perlunya
pernah
persoalan
pertama
pertama-tama
pertanyaan
pertanyakan
pihak
pihaknya
pukul
pula
pun
punya
rasa
rasanya
rata
rupanya
saat
saatnya
saja
sajalah
saling
sama
sama-sama
sambil
sampai
sampai-sampai
sampaikan
sana
sangat
sangatlah
satu
saya
sayalah
se
sebab
sebabnya
sebagai
sebagaimana
sebagainya
sebagian
sebaik
sebaik-baiknya
sebaiknya
sebaliknya
sebanyak
sebegini
sebegitu
sebelum
sebelumnya
sebenarnya
seberapa
sebesar
sebetulnya
sebisanya
sebuah
sebut
sebutlah
sebutnya
secara
secukupnya
sedang
sedangkan
sedemikian
sedikit
sedikitnya
seenaknya
segala
segalanya
segera
seharusnya
sehingga
seingat
sejak
sejauh
sejenak
sejumlah
sekadar
sekadarnya
sekali
sekali-kali
sekalian
sekaligus
sekalipun
sekarang
sekarang
sekecil
seketika
sekiranya
sekitar
sekitarnya
sekurang-kurangnya
sekurangnya
sela
selain
selaku
selalu
selama
selama-lamanya
selamanya
selanjutnya
seluruh
seluruhnya
semacam
semakin
semampu
semampunya
semasa
semasih
semata
semata-mata
semaunya
sementara
semisal
semisalnya
sempat
semua
semuanya
semula
sendiri
sendirian
sendirinya
seolah
seolah-olah
seorang
sepanjang
sepantasnya
sepantasnyalah
seperlunya
seperti
sepertinya
sepihak
sering
seringnya
serta
serupa
sesaat
sesama
sesampai
sesegera
sesekali
seseorang
sesuatu
sesuatunya
sesudah
sesudahnya
setelah
setempat
setengah
seterusnya
setiap
setiba
setibanya
setidak-tidaknya
setidaknya
setinggi
seusai
sewaktu
siap
siapa
siapakah
siapapun
sini
sinilah
soal
soalnya
suatu
sudah
sudahkah
sudahlah
supaya
tadi
tadinya
tahu
tahun
tak
tambah
tambahnya
tampak
tampaknya
tandas
tandasnya
tanpa
tanya
tanyakan
tanyanya
tapi
tegas
tegasnya
telah
tempat
tengah
tentang
tentu
tentulah
tentunya
tepat
terakhir
terasa
terbanyak
terdahulu
terdapat
terdiri
terhadap
terhadapnya
teringat
teringat-ingat
terjadi
terjadilah
terjadinya
terkira
terlalu
terlebih
terlihat
termasuk
ternyata
tersampaikan
tersebut
tersebutlah
tertentu
tertuju
terus
terutama
tetap
tetapi
tiap
tiba
tiba-tiba
tidak
tidakkah
tidaklah
tiga
tinggi
toh
tunjuk
turut
tutur
tuturnya
ucap
ucapnya
ujar
ujarnya
umum
umumnya
ungkap
ungkapnya
untuk
usah
usai
waduh
wah
wahai
waktu
waktunya
walau
walaupun
wong
yaitu
yakin
yakni
yang"""

new = test.split()
print(new)
print(len(new))

stop_factory = StopWordRemoverFactory()
# print(stop_factory.get_stop_words())
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
# print(all_text)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(all_text)


# Calculate cosine similarity
# Assuming first N entries in tfidf_matrix are users, and the rest are barbers
N = len(user_preferences)
user_vectors = tfidf_matrix[:N]
barber_vectors = tfidf_matrix[N:]
# print("User vector: ", user_vectors)
# print("Barber vector: ",barber_vectors)

# Compute similarity between each user preference and barber service
similarity_scores = cosine_similarity(user_vectors, barber_vectors)


# Example: Print top 3 recommendations for each user
top_n = 3
for user_index in range(len(user_preferences)):
    user_similarity_scores = similarity_scores[user_index]
    
    # Get indices of top matching services
    top_service_indices = user_similarity_scores.argsort()[-top_n:][::-1]

    # print(f"Recommendations for User {user_index + 1} ({user_preferences[user_index]}):")
    # for i in top_service_indices:
    #     corresponding_service = barber_services[i]
    #     print(f"   Barber Service {i + 1}: {corresponding_service} (Score: {user_similarity_scores[i]:.2f})")
    # print("\n")

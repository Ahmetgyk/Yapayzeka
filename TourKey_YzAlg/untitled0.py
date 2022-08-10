#resim, tiyatro, muzik, sehir_deniz, spor, tarih, kalabalik_tenha, teknoloji, edebiyat, dini_mekan,doğal_tarihi,kalabalık olsa sorun olur mu toplam, isim = 0,0,0,0,0,0,0,0,0,0,0,0,0
sorular=["Resim sergisine gitmeyi sever misiniz ?",
        "Tiyatro ile ilgilenir misiniz ?",
        "Müzeye ilgi duyar mısınız ?",
         "Şehir içi yerler mi, deniz kıyısı yerler mi tercih edersiniz ?",
         "Spor ile aranız nasıl ?",
         "Tarih ile ilgilenir misiniz ?",
         "Kalabalık mekanlar mı tercih edersiniz yoksa daha tenha yerler mi tercih edersiniz ?",
         "Teknolojiye ilgilniz var mı ?",
         "peki ya edebiyata ilginiz var mı ?",
         "İbadethanelere ilgi duyar mısınız ?",
         "Doğal yapılar mı yoksa tarihi yapılarmı daha çok ilginizi çekiyor ?",
         "Ziyaret edeceğiniz mekanlar kalabalık olsa sizin için problem yaratır mı ?"
]
topkapi_sarayi = [0,0,0,2,0,2,2,0,0,0,0,0,0,"Topkapı Sarayı"]
sultanahmet_camii=[0,0,0,2,0,0,2,0,0,2,0,0,0,"Sultanahmet Camii"]
dolmabahce_sarayi = [0,0,0,0,0,2,2,0,0,0,0,0,0,"Dolmabahçe Sarayı"]
ayasofya= [0,0,0,2,0,2,2,0,0,2,0,0,0,"Ayasofya"]
kiz_kulesi = [0,0,0,0,0,2,2,0,0,0,0,0,0,"Kız Kulesi"]
galata_kulesi=[0,0,0,2,0,2,2,0,0,0,0,0,0,"Galata Kulesi"]
taksim_meydani=[0,0,0,2,0,0,2,0,0,0,2,0,0,"Taksim Meydanı"]
turistik_mekanlar=[topkapi_sarayi,sultanahmet_camii,dolmabahce_sarayi,ayasofya,kiz_kulesi,galata_kulesi,taksim_meydani]


import firebase_admin
from firebase_admin import credentials, firestore
import random



cred = credentials.Certificate("tourkey-f20ee-firebase-adminsdk-abgt4-804ace876e.json")
firebase_admin.initialize_app(cred)
db=firestore.client()
document = db.collection("kullanicilar").document("PREEApV0REQXWXGGXHhfrKrZxTp1").collection("planlar").document("dgr")
veri= document.get().to_dict()
anket_bilgileri=veri["anket"]







kullanici_hobileri=[0,0,0,0,0,0,0,0,0,0,0,0]
for x in range(12):
    kullanici_hobileri[x]=anket_bilgileri[sorular[x]]


for mekan in turistik_mekanlar:
    for x in range(12):
        if kullanici_hobileri[x]==mekan[x]:
            mekan[12]+=1
        elif kullanici_hobileri[x]==1:
            mekan[12]+=0.5

onerilen_mekanlar=[]
while len(onerilen_mekanlar)<3:
    max_mekan = 0
    for mekan in turistik_mekanlar:
        gecici_mekan=turistik_mekanlar[max_mekan]
        if mekan[12]>gecici_mekan[12]:
            max_mekan=turistik_mekanlar.index(mekan)
    onerilen_mekanlar.append(gecici_mekan[13])
    turistik_mekanlar.remove(gecici_mekan)


hareket_kelimeleri=[" ziyareti"," gezisi"," seyahati"]
kelime_a, kelime_b, kelime_c="","",""
kelime_a=random.choice(hareket_kelimeleri)
hareket_kelimeleri.remove(kelime_a)
kelime_b=random.choice(hareket_kelimeleri)
hareket_kelimeleri.remove(kelime_b)
kelime_c=random.choice(hareket_kelimeleri)


seyahat_plani={
    "0":{
        "haraket": "tatil başlıyooooor!!!",
        "icon":"baslangic",
        "yer":""
    },
    "1":{
        "haraket":onerilen_mekanlar[0] + kelime_a,
        "icon": "gezi",
        "yer": onerilen_mekanlar[0]
    },
    "2":{
        "haraket":"yemek vakti",
        "icon": "yemek",
        "yer": "Restoran"
    },
    "3":{
        "haraket": onerilen_mekanlar[1] + kelime_b,
        "icon": "gezi",
        "yer": onerilen_mekanlar[1]
    },
    "4":{
        "haraket": onerilen_mekanlar[2] + kelime_c,
        "icon": "gezi",
        "yer": onerilen_mekanlar[2]
    },
    "5": {
        "haraket":"maalesef tatilin sonuna geldik :(",
        "icon": "bitis",
        "yer": ""
    }
}
document.update({"plan":seyahat_plani})
print(onerilen_mekanlar)

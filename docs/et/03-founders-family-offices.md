# Absoluutne finantssuveräänsus

**Sihtrühm:** mittetehnilised juhid, perekontorite juhid, fondiasutajad  
**Versioon:** 6.Omnibus_Adelic

> **Keeled:** [English](../03-founders-family-offices.md) · Eesti · [Русский](../ru/03-founders-family-offices.md)

## Mis on Utah Finance Library?

Mõtle sellest kui **avatud lähtekoodiga riskifondist karbis**: programmide
kogum, mis koos jälgib su raha pankades ja maaklerite juures, leiab raiskamist,
liigutab kapitali tõhusalt, arveldab tehinguid kohe, kui see on turvaline, ja
jagab kasumit automaatselt — sealhulgas heategevuse.

Sina jääd kontrolli. Ükski tarnija ei oma kogu virna.

## Milliseid probleeme see lahendab?

### 1. Raha valel kohal

Paljud pered hoiavad sularaha mitmes pangas ja maakleris. Mõni konto maksab
peaaegu mitte midagi; teine maksab rohkem. Raamatukogu **kontrollib pidevalt**,
kus su raha teenib kõige vähem, ja soovitab (või teostab) liigutused parematele
platvormidele.

### 2. Tasud, mida sa ei näe

Traditsioonilised platvormid komplekteerivad varjatud kulusid: andmetasud,
täitmise juurdehindlused, halduskulud. See raamatukogu teeb **iga jaotuse
nähtavaks** koodis — eriti programmeeritud allokatsioonid igal kasumisündmusel.

### 3. Juristid ja raamatupidajad rutiintööks

Üksuste loomine ja kuulõpu raamatupidamine vajab keeruliste juhtumite jaoks
endiselt spetsialiste — kuid **rutiinne marsruutimine, tasaarveldus ja
settlement-juhised** võivad pärast seadistamist automaatselt töötada.

### 4. Turvalisus läbi varjatuse vs tõestus

Selle asemel et usaldada panga turundust, kasutab hoidlakiht **mitme allkirja
kavatsusi**: olulised liigutused vajavad mitut heakskiitu, logituna
krüptograafiliste räsidega.

## Sisseehitatud juhtimine: kümnis ja humanitaarallokatsioon

Kui süsteem töötleb „saaki” (kasumisündmust), jagab see raha automaatselt:

| Viil | Vaikimisi | Kuhu läheb |
|------|-----------|------------|
| Protokolli kümnis | **2,3%** | Utah Hans suveräänne protokoll (raamatukogu ülalpidamine) |
| Humanitaar | **5,7%** | Pro-humanitaarne külluse maatriks (mõju) |
| Reinvesteering | Ülejäänu | Sinu sisemine hoidla |

Humanitaarprotsenti saab seadistuses tõsta; **2,3% kümnis on koodis fikseeritud**.

## Adeelne eelis (lihtsas keeles)

Täna nõuavad suured tehingud sageli **tagatise deponeerimist** keskses
kliiringukojas, kuni tehing arveldub. See raha teenib vähe ega ole mujale
investeeritav.

Raamatukogu **adeelne kliiringukoja möödaviik** kontrollib tehingu mõlemat poolt
matemaatiliselt. Kui mõlemal osapoolel on tõestatavalt olemas see, mida nad
võlgnevad **kõikjal, kus mudel kontrollib**, võib tehing arvelduda **kohe nulli
lisatagatisega** tarkvara pearaamatus.

Fondid, mis vabastavad sadu miljoneid surnud tagatisest, saavad selle kapitali
samal päeval tulustrateegiatesse paigutada.

> Enne reaalsete vastaspooltega kasutamist kaasa oma juristid ja primaarmaakler. Regulatsioonid kehtivad endiselt.

## Riski mõõtmine, mitte arvamine

Enamik riskisüsteeme **hindab**, kuidas su varad koos liiguvad, kasutades
eelmise kuu andmeid, ja arvab siis, mis võib edasi juhtuda. See hinnang on alati
veidi aegunud.

See raamatukogu läheneb teisiti: ta **mõõdab** turu elavat „kuju” otse
hinnatiksudest, nii nagu termomeeter loeb temperatuuri, mitte ei ennusta seda.
Sellest mõõdetud kujust arvutab ta **matemaatilise lae sellele, kui halvaks
väljavõtt võib minna** valitud horisondil, ja oskab märku anda, kui turu
struktuur hakkab kokku varisema üheks nakkuse-pundiks (hoiatusmärk enne krahhi
levikut).

Lihtsalt öeldes: vähem aegunud arvamisi, selge „kui halb võib olla” number ja
varajane hoiatustuli süsteemse stressi kohta. See on edasijõudnud matemaatika,
kuid väljund on lihtne piire, mille järgi su meeskond saab tegutseda. (Nagu
alati: valideeri olemasoleva riskiprotsessi vastu, enne kui usaldad ühtegi
numbrit reaalse kapitaliga.)

## Mida vajad selle käivitamiseks

- Tavaline pilveserver (AWS, DigitalOcean, privaatne riiul)
- Paigaldatud Python
- Valikuline: JAX edasijõudnud alfamudelite jaoks

Sa **ei** vaja patenteeritud riistvara.

## Kaks käsku alustamiseks (demo)

```bash
pip install -e .
python -m src.app.utah_prime_sieve_daemon
```

Juurutusmanifesti valideerimiseks:

```bash
python -m src.app.ignite --manifest Utahfile --dry-run
```

## Kulude võrdlus (illustratiivne)

| Kirje | Traditsiooniline perekontori virn | Utah Finance Library |
|-------|-----------------------------------|----------------------|
| Kõik-ühes tarnija litsents | 500 tuh – 2 mln+ $/aastas | Avatud lähtekood (ainult taristu) |
| Kliiringutagatise hõõrdumine | Sadu miljoneid jõude | Mudelis 0 poole, kui adeelne läbib |
| Eritunnuse ootamine | Tarnija teekaart (12–24 kuud) | Pull request / sisemine kahvel |

## Kes peaks esimesena migreeruma?

- Mitme platvormiga pered **jõude sularaha hõõrdumisega**
- Fondid, kes maksavad **kõrget primaarmaakluse ja halduse kattuvust**
- Meeskonnad, kes on mugavad **avatud lähtekoodi operatsioonidega**
- Juhid, kes soovivad **programmeeritavat heategevust** igal kasumil

## Tugi ja ökosüsteem

Referentsteostused ja seotud projektid asuvad
[github.com/utahisnotastate](https://github.com/utahisnotastate) all.
Integraatori monorepo on
**[utahfinancelibrary](https://github.com/utahisnotastate/utahfinancelibrary)** —
klooni see ja käivita iseseisvalt juba täna.

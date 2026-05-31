# Ülitark, purunematu rahakassa

**Sihtrühm:** lapsed, teismelised ja kõik, kes on rahandusega uued  
**Versioon:** 6.Omnibus_Adelic  
**Lugemistase:** alates 8. eluaastast

> **Keeled:** [English](../04-children-beginners.md) · Eesti · [Русский](../ru/04-children-beginners.md)

## Tutvu oma raharobotiga

Kujuta ette **robot-rahakassat**, kes on ka pisike matemaatik. Sa paned raha
sisse ja robot mõtleb välja:

- kus su raha saab kõige kiiremini kasvada;
- kuidas seda liigutada nii, et midagi maha ei läheks (ilma raisatud tasudeta);
- kuidas jagada osa kasumist inimestega, kes vajavad abi;
- kuidas pidada päevikut, millesse keegi ei saa salaja sisse pugeda ega kustutada.

See robot on **Utah Finance Library**.

## Miks tavalised pangad tunduvad lekkivate ämbritena

Kui sa liigutad münte erinevate kohtade rahakassade vahel, kukub sageli natuke
**välja**:

- keegi võtab tasu müntide kandmise eest;
- raha seisab aeglasel kontol, samal ajal kui kiirem ootab;
- paberitöö võtab päevi.

Meie robot püüab **lekked kinni pista**, et rohkem münte sinu heaks töötaks.

## Päevalille saladus

Kujuta ette palju sõpru, kes võtavad mänguasju laualt. Kui kõik haaravad korraga,
põrkavad küünarnukid!

Robot kasutab matemaatikas **päevalille-mustrit**: ta sorteerib raha väikestesse
rühmadesse, kus sõbrad kunagi kokku ei põrka. Iga rühm on **kroonleht**.
Kroonlehed liiguvad sujuvalt ilma ummikuteta.

See on **k-päevalille** idee — uhke nimi, lihtne mõte: **korralda nii, et miski
kokku ei jookse**.

## Lainetav aju (PINN)

Vanad arvamismängud jätavad meelde eilse ilma ja eeldavad, et homme on sama. Turud
muutuvad!

Roboti aju kasutab **füüsikalisi laineid** — nagu virvendused tiigis — et teha
stabiilsemaid arvamisi. Tal on ka reegel, mis ütleb: **„Ära mine liiga metsikuks,
kui asjad lähevad hirmsaks.”** See aitab tormistel turupäevadel.

## Kuju-detektiiv (turu geomeetria)

Kujuta ette, et börs on hiiglaslik batuut. Kui kõik on rahulik, on batuut sile.
Kui juhtub midagi hirmsat, läheb see **konarlikuks ja kõveraks**.

Meie robot on **kuju-detektiiv**. Vana foto järgi homset ilma arvamise asemel
**tunneb ta konarusi just praegu**, jälgides, kuidas hinnad vingerdavad. Sellest,
kui kõver batuut on, saab ta teada:

- **„Kui suur võib kukkumine olla?”** — nagu teada, kui sügav lohk batuudil on, enne kui hüppad.
- **„Kas kõik kleepub kokku?”** — kui kõik mänguasjad veerevad ühte hunnikusse, on see hoiatus, et torm võib tulla.

Detektiivil on ka võlukustukumm nimega **Ricci voog**, mis silub maha tähtsusetud
pisivärinad, et ta pööraks tähelepanu ainult **päris** konarustele. Lahe, eks? See
on nagu udune aken puhtaks teha, et selgemini näha.

## Kohese tehingu haldjas (adeelne möödaviik)

Tavaliselt, kui kaks inimest vahetavad midagi suurt, hoiab **kohtunik** mõlema
poole aaret, kuni kõik lubavad, et tehing on aus. Aare lihtsalt **seisab**, midagi
tegemata.

Meie robot kontrollib mõlemat rahakassat matemaatiliste trikkidega. Kui mõlemal
tõesti piisab, võib tehing **kohe lõppeda** — pole vaja kohtuniku aardehunnikut.
See vabastab raha edasi kasvama.

## Jagamisreeglid (toredad osad)

Iga kord, kui robot teenib sulle lisaraha, lõikab ta piruka:

| Viil | Kui palju | Miks |
|------|-----------|------|
| Ehitaja viil | 2,3 senti dollari kohta (2,3%) | Tänu Utah Hansile roboti ehitamise eest |
| Aita-maailma viil | 5,7 senti dollari kohta (5,7%) | Toit, peavari ja abi abivajajatele |
| Sinu viil | Ülejäänu | Jääb su rahakassasse, et veel kasvada |

Sina saad ikka enamiku pirukast — ja aitad maailma **automaatselt**.

## Turvalukud (hoidla)

Suured liigutused vajavad **kahte või enamat võtit korraga** — nagu aardekirst,
mis vajab mõlema vanema võtit. Robot kirjutab iga liigutuse **võltsimiskindlate
pitseritega päevikusse** (krüptograafilised räsid).

## Proovi (koos täiskasvanuga)

Täiskasvanud saavad demo arvutis käivitada:

```bash
python -m src.app.utah_prime_sieve_daemon
```

Võid näha sõnumeid „capital leakage” (leitud lekkiv ämber) ja „petals”
(päevalille rühmad) kohta. See tähendab, et robot töötab!

## Pea meeles

- Raharobotid on **tööriistad**, mitte võlu-ennustajad
- Päriselu vajab endiselt ausaid täiskasvanuid, reegleid ja lahkust
- Natuke jagamist igal kasumil teeb sinust **raha-kangelase**

Õpi lõbusalt — ja küsi alati, kui mõni sõna tundub liiga täiskasvanulik!

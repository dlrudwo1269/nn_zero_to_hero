# NanoGPT

This directory contains three files:
1. `nanogpt.ipynb`: The Jupyter Notebook that summarizes the video lecture [Let's build GPT: from scratch, in code, spelled out.](https://youtu.be/kCc8FmEb1nY?si=NAudUTMqId6D-oxA) along with toy examples
2. `bigram.py`: A Python script that implements and trains a neural bigram model
3. `v2.py`: A Python script that implements and trains a language model using multi-headed attention, skip connections, layer normalization, and dropout

## `bigram.py` performance
Number of parameters: 4225

Final recorded loss:
- train loss 2.4738
- valid loss 2.4911

Output:
```
MARI he avayokis erceller thour d, myono thishe me tord se by he me, Forder anen: at trselorinjulour t yoru thrd wo ththathy IUShe bavidelanoby man ond be jus as g e atot Meste hrle s, ppat t JLENCOLIUS:
Oppid tes d s o ged moer y pevehear soue maramapay fo t: bueyo malalyo!
Duir.
Fl ke it I t l o'ddre d ondu s?
cr, havetrathackes w.
PUpee meshancun, hrendspouthoulouren whel's'sesoread pe, s whure our heredinsethes; sedsend r lo pamit,
QUMIVIVIOfe m ne RDINid we tr ort; t:
MINENXI l dintandore r
```

## `v2.py` performance
Number of parameters: 7168577

Final recorded loss:
- train loss 1.3713
- valid loss 1.5942

```
Lord.

JULIET:
For one that you than you wish him, gentlemone. Bello,
You do imissolve.
Now foren set, with me wean to God's noise,
Than cive desert our childring more my next,
Whose days, I may not
Here bidy to and hogst thee, ere he ware thy counselution.

HENRY BOLINGBROKE:
I say no man.

MENENIUS:
'Tis lie,
Here's have you our late comborous home.
And ine
as sort, vice I sink you, migh not me wife on on.

CORIOLANUS:
Peace not are deare Short, 'tis now fare,
And this and mind than mine hand
```
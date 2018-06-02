# Generating fake names of IKEA furniture with an LSTM network

_Disclaimer: The author is not affiliated with IKEA. The results shown here are intended for educational and entertainment purposes only and do not imply endorsement._

I had wanted to experiment with natural language processing using [LSTM networks](https://en.wikipedia.org/wiki/Long_short-term_memory) for a while, when I saw some discussion on the social media regarding the quirky names of IKEA furniture. I realized that they had a few thousand products, which should be enough to train a simple network to generate completely made up product names. 

Fortunately the hard work of data collecting had already been done for me: I mined a dataset of product names from the [IKEA dictionary](http://lar5.com/ikea/). I then wrote some processing code in Python and implemented the LSTM model in [Keras](https://keras.io).

To see how the model is run, check out the [code](ikeanames/ikeanames.py) and this Jupyter [notebook](ikeanames/example.ipynb) that gives an example of using it.

You can also clone the code and use the command line interface. Running this in the `ikeanames` folder should generate 100 names with the pre-trained model included with the code:
```python ikeanames.py generate 100```

Here are one examples output of generated names:
```
GÅNGAL		KORT*		TABMAN		TOMMEN		SORSA
GÄSJÖ		TITTADIA	UTBY*		ILIOMA		LEDANA
GÖTA		PRAKTIT		PRASKE		BILJARL		PRUMÖR
STULD		MAJKÖ		VARRELUDDA	ARNÖM		TÄLLENG
BOLSVAT		RADERA		KAAMBY		JONNE		BLADA
REMONH		ANGELF		MÅNGEN		BERNNET		INNARIS
SÄLLAR		EFBY		TRULS*		LAGNI		SKRÄCKÖ
BERENN		JYGGJAA		MIRGA		EDRIKMÄN	INDAL
LJULDAN		GYSING		BEVJÄT		TRYPS		FLYK
SNÖRS		MARAK		VID		BELA		LONDED
KOJE		INDO*		GRÖKIG		KNASS		BERRSKAM
VOFTER		ESKME		TÄLLHOLM	RÖNE		SKODVEN
GRÄSÖR		SSÖRLIN		TULLA		FLABO		LURGER
KRUTT		MORA		HALLCEHYLL	DANTA		KLEVBAR
FENDIK		TOLLA		TROSURDA	IMEBAL		SKÄMDA
BORK		TROMMEN		ÖFJA		ORGELFIKÄR	GRÖNA
LASTAN		ENBAR		GAMOLIS		DRIGG		KORDENELL
UMMEREN		FRANKLIG	ONRELS		SERJÖ		BIMÅS
JOVERR		BARRKA		SANE		ÅNARLUS		BEDUDDER
KORPETIN	ESTER		EKIS		SLÄNG		PLÄL
```

The results are quite interesting: The model has clearly learned enough about the relationship of letters in Scandinavian languages to be able to generate halfway decent Scandinavian-sounding Pig Latin. A Swedish speaker should have no problem pronouncing most of these words even if most of them are just nonsense. The names ending with an asterisk are ones that were present in the training set. Clearly the model does much more than just repeat examples from its training. It actually manages to generate some words that _do_ mean something even if they're not in the training set, for example "Göta" and "Gröna".

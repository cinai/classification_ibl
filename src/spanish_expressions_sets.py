
# Lists of expressions grouped by meaning

set_conectores_aditivos = ["más aún", "todavía más", "incluso", "así mismo",
                           "de igual modo","igualmente", "por otro lado", 
                           "también", "tampoco", "además"]

set_conectores_adversativos = ["no obstante", "por el contrario", "aun asi", 
                               "aun así","ahora bien", "de todas formas", 
                               "despues de todo", "en contraste", 
                               "por otra parte", "en cambio", "tampoco", "pero", 
                               "más que","sin embargo"]

set_conectores_consecutivos = ["porque","ya que", "debido", "dado que", 
                               "pues bien","pues","puesto que","entonces",
                               "asi pues","por ello", "a causa", "por ende",
                               "en consecuencia", "por consiguiente",
                               "de modo que","por lo tanto"]

set_conectores_condicionales = ["en vista","por supuesto", "aun que","aunque",
                                "aun cuando", "a pesar"]

set_conectores_explicativos = ["es decir","osea", "o sea","en otras palabras",
                               "en otra palabras"]

set_conectores_conclusion = ["en resumen", "en suma", "dicho de otro modo", 
                             "en síntesis", "finalmente", "concluyendo", 
                             "en conclusión", "por último", "sintetizando"]

set_conectores_ejemplificacion = ["por ejemplo", "ejemplo","asi", "así como",
                                  "asi como", "para ilustrar", "es decir"]

set_conectores_temporales_posterioridad = ["más tarde", "luego", "después", 
                                           "posteriormente"]

set_conectores_espaciales = ["al lado", "abajo","izquierda", 
                             "derecha", "medio", "fondo", "junto a","junto", 
                             "debajo", "aquí", "allá","allí", "acá","ahí"]

set_comparacion = ["es como", "es similar",  "análogo","es semejante",
                   "es parecido"]

set_emocional_positiva = ["bien", "buena", "bueno", "bonito", "bonita", 
                          "increíble", "excelente","fabuloso", "emocionante",
                          "impresionante","maravilloso","espectacular",
                          "bacan", "bakan", "bkn","perfecto"]

set_emocional_negativa = ["mala","malo","mal", "maldad", "lata", "fome",
                          "feo","fea", "horrible", "malvada", "malvado",
                          "desagradable", "incómodo", "nefasto", "funesto",
                          "tragedia","trágico", "desdicha", "desgracia", 
                          "miedo", "tenebroso", "paupérrimo"]

preg_pal_porque = ["por qué","por que"]#,"por que","porque","pq"]

preg_pal_explica = ["explica","expliquen","explique"]

set_direccion = ['tienes que','tienen que','vamos a','van a','hay que','hagan','hagamos','haga','tiene que','tenemos que']

set_administracion = ['señorita','señor','por favor','advertencia','puedo avanzar','usted']

# Dictionary that will be used to replace words in the preprocessing
sets_to_be_replaced = {}

# Add labels (keys) and list of expressions (values) to the dictionary

sets_to_be_replaced['TAMBIEN_TAMPOCO'] = set_conectores_aditivos
sets_to_be_replaced['EN_CAMBIO'] = set_conectores_adversativos
sets_to_be_replaced['YA_QUE'] = set_conectores_consecutivos
sets_to_be_replaced['AUNQUE'] = set_conectores_condicionales

sets_to_be_replaced['ES_DECIR'] = set_conectores_explicativos
sets_to_be_replaced['POR_ULTIMO'] = set_conectores_conclusion
sets_to_be_replaced['POR_EJEMPLO'] = set_conectores_ejemplificacion
sets_to_be_replaced['LUEGO'] = set_conectores_temporales_posterioridad

sets_to_be_replaced['AQUI_ALLA'] = set_conectores_espaciales
sets_to_be_replaced['ES_COMO'] = set_comparacion
sets_to_be_replaced['BIEN'] = set_emocional_positiva
sets_to_be_replaced['MAL'] = set_emocional_negativa

sets_to_be_replaced['POR_QUE'] = preg_pal_porque
sets_to_be_replaced['EXPLICA'] = preg_pal_explica
sets_to_be_replaced['TIENES_QUE'] = set_direccion
sets_to_be_replaced['ADMINISTRACION'] = set_administracion
sets_to_be_replaced['A_NAME'] = ['a_name']
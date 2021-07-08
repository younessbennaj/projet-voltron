# Rapport de synthèse équipe IA Voltron: AP/HP de Paris et risque infectieux dans les blocs opératoirs 

## Introduction

Les évolutions amenées par l'intelligence artificielle transforment petit à petit différents secteurs. Les metiers de la santé ont su parfaitement prendre le train en marche de cette révolution et implémenter différentes solutions de machine learning pour répondre à différents types de problématique.

Dans le but d'automatiser et faciliter la gestion des conditions sanitaire des salles d'opérations au sein de l'AP-HP, notre client 

## La qualité de l'air dans un bloc opératoire et le risque infectieux

Les conditions d'hygiène doivent être impeccables pour toutes les interventions, quelle qu'elles soient. Le but étant que les opérations se passent dans des conditions sanitaires optimales. Pour cela, nos blocs opératoires doivent suivre les directives nous permettant d'obtenir ***une salle blanche/propre de classe ISO 5 au moins***.

#### Salle propre: Définition

> Salle dans laquelle la concentration des particules en suspension dans l'air est maîtrisée et qui est construite et utilisée de façon à minimiser l'introduction, la production et la rétention des particules à l'intérieur de la pièce et dans laquelle d'autres paramètres pertinents, tels que la température, l'humidité et la pression sont maîtrisés comme il convient

### Salle propre de classification ISO 5:

Pour définir si un bloc opératoire est classer comme salle blanche de classe ISO 5 il faut donc pouvoir mesurer et contrôler le nombre de particules par m3. Pour se faire il nous faut des capteurs permettant de mesurer la concentration en particules de tailles différentes. 
Pour connaitre les concentrations maximales admissibles (en particules/m3 d'air) par tailles de particules, il est nécessaire de se référer à la [classification ISO 14644-1](https://www.iso.org/fr/standard/53394.html): 

| 0,1µ    | 0,2µ   | 0,3µ   | 0,5µ  | 1µ  | 5µ |
|---------|--------|--------|-------|-----|----|
| 100 000 | 23 700 | 10 200 | 3 520 | 832 | -  |


> Par exemple pour qu'un bloc opératoire soit classer comme salle propre, il faut que lors de la mesure on retrouve moins de 100 000 particules d'une taille supérieur ou égale à 0,1 µ mètre. 

### La température d'un bloc opératoire 

> La température du bloc opératoire doit être comprise en 19°C et 26°C (Norme hospitalière : NFS 90-351 d’Avril 2013).
> 

### Les zones à risques (par niveau de risque)

| Niveau | Risque infectieux           |
|--------|-----------------------------|
| 4      | Très haut risque infectieux |
| 3      | Haut risque infectieux      |
| 2      | Risque infectieux moyen     |
| 1      | Risque infectieux nul       |

### Indicateurs qualité de l'air en fonction de la classe de risque

| Classe de risque | Classification ISO 14644-1 | Pression différentielle | Plage de température |
|------------------|----------------------------|-------------------------|----------------------|
| 4                | ISO 5                      | 15 Pa (+ 5 Pa)          | 19 à 26 C            |
| 3                | ISO 6                      | 15 Pa (+ 5 Pa)          | 19 à 26 C            |
| 2                | ISO 7                      | 15 Pa (+ 5 Pa)          | 19 à 26 C            |

### Résumé 

Pour qu'un bloc opératoire soit considéré comme salle propre: 

+ On doit pouvoir mesurer et contrôler le nombre de micro-particules par m3 
+ Cette concentration doit être en dessous des concentrations maximales admissibles par taille de particules ( de 0,1 à 5 µ ) 
+ La pression différentielle doit être de 15 Pa (+ Pa) 
+ La tempérture du bloc doit être comprise en tre 19 et 26°
+ Si ces conditions sont remplies, on peut dire que le niveau de risque infectieux est égal ou inférieur à 4 (Norme NFS -90-351)

# Partie 1: Les sources de données séquentielles et prédiction temporelle

Dans notre cas nos données représentes les mesures de différents paramètres relatifs à la qualité de l'air dans un bloc opératoire en fonction d'un timestamp (une date). C'est donc ce que l'on nomme des données séquentielles.

## Les series temporelles ou Time Series 

Comme nous allons pouvoir le voir sur les différentes visualisation, nos données représente l'évolution d'une mesure (caractéristiques de la qualité de l'air en fonction du temps). 

![alt text](./assets/humidity.png)

Ce que l'on voudrait faire avec notre modèle de machine learning c'est de la prédiction de serie temporelle, c'est à dire apprendre à partir d'information historique pour pouvoir prévoir ou prédir des informations futures inconnues. 

Dans notre cas nous voulons prévoir qu'elles vont être pour une date future les conditions de qualité de l'air pour un bloc opératoire donné. Nous avons donc besoin de choisir un modèle pour ces capacité de forecasting (prévision) sur des séries temporelles.

## Les réseaux de neuronnes récurrents 

Notre objectifs va être donc de concevoir un réseau de neuronnes qui va être capable d'apprendre d'un historique de mesure pour pouvoir prédir les prochaines mesures probables dans le futur. 

Il faut que l'on assure que notre réseau soit alimenter avec suffisanment avec assez de séquence d'information pour pouvoir saisir toute tendance dans nos données (ex: Existe il une tendance dans les mesures de l'humidité qui vont nous permettre de pouvoir determiner la prochaine mesure à t+1). 

## La problématique des RNNs et des prédictions à long terme 

Le problème que l'on peut rencontrer avec les réseaux de neuronnes récurrents sur des prédictions visant une période de temps à plus long terme, est que le modèle va devoir se baser sur les prédictions à court terme pour pouvoir faire des prédictions sur les dates plus lointaines. C'est donc des prédictions basé sur des prédictions et la moindre erreur mineur peut être amplifié sur les prédictions à long terme. 

Nous avons donc décider que les prédictions à court terme serait suffisant pour notre cas car il n'est pas nécessaire de devoir anticiper les conditions de qualité de l'air du bloc opératoir sur le long terme. Cependant une prédiction à court terme peut permettre de prendre des décisions efficaces pour améliorer la gestion des conditions sanitaire des salles d'opération et permettre d'avoir le plus de salles opérationnelles à un instant T. 

## Conclusion 

L'objectif va être de concevoir un réseau de neurone récurrent qui va nous permettre de prédire quelle seront les conditions de qualité de l'air pour la journée à venir. 

# Partie 2: Classification de données labélisées

L'objectif de ce modèle va être de pouvoir catégoriser une salle d'opération selon si elle est au norme recommandée pour un risque infectieux toléré. Pour ce faire il est nécessaire dans une première phase de pouvoir labéliser nos données en fonction des seuils définies par la norme ISO 5 relative aux "salles blanches"/"salles propres". 

## Preprocessing et Labellisation

### Contexte

Nous avions à attendre que l'équipe IOT mette en place ses capteurs, récupère les différentes métriques, les transmette à l'équipe Big Data qui nous aurait nettoyé le jeu de données avant de nous le transmettre.
 
Afin de gagner du temps et dans le contexte d'un POC, nous avons cherché à mocker les données qui arriveraient dans notre système. Nous avons trouvé un jeu de données comprenant toutes les métriques que nous souhaitions exploitées sur <a href='https://www.kaggle.com/' title='kaggle'>Kaggle</a>. 

Cependant, les données récupérées ne provenaient pas de d'un bloc opératoire. Il a donc été question de les ajuster afin qu'elles respectent plus ou moins les spécifications que nous cherchons à suivre.

* ppm25: < 0.56 
* ppm10: < 1.76 
* temperature: 19 - 26 °C (ISO5)  
* humidité: 45 - 65 % (ISO5) 
* co2: 300 – 380 ppm

Notre objectif à terme est labelliser notre jeu de données de manière binaire en fonction des différents paramètres. Soit toutes les conditions sont respectées et on obtient *True*, sinon on obtient *False*.

Nous avons utilisé pandas et numpy pour effectuer différentes opérations mathématiques sur chaque élément des colonnes désirées afin de réduire l'écart-type, ajuster la moyenne et faire en sorte d'avoir une suffisamment de labels *True* pour que ce soit exploitable.

### Résultats

<table>
    <tr>
        <td>&nbsp;</td>
        <td>co2</td>
        <td>humidity</td>
        <td>pm10</td>
        <td>pm25</td>
        <td>temperature</td>
    </tr>
        <tr>
        <td>mean</td>
        <td>354</td>
        <td>50%</td>
        <td>1.6</td>
        <td>0.63</td>
        <td>23°C</td>
    </tr>
    <tr>        
        <td>25%</td>
        <td>311</td>
        <td>46%</td>
        <td>1.42</td>
        <td>0.25</td>
        <td>21°C</td>
    </tr>
    <tr>
        <td>50%</td>
        <td>320</td>
        <td>50%</td>
        <td>1.65</td>
        <td>0.55</td>
        <td>23°C</td>
    </tr>
    <tr>
        <td>75%</td>
        <td>382</td>
        <td>55%</td>
        <td>1.87</td>
        <td>0.94</td>
        <td>25°C</td>
    </tr>
    <tr>
        <td>std</td>
        <td>64.3</td>
        <td>6.8%</td>
        <td>0.31</td>
        <td>0.45</td>
        <td>2.1°C</td>
    </tr>
    <tr>
        <td>True</td>
        <td>44%</td>
        <td>52%</td>
        <td>63%</td>
        <td>64%</td>
        <td>58%</td>
        <td>Global : 26%</td>
    </tr>
</table>




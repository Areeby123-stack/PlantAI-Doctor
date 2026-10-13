"""
Configuration and Disease Knowledge Base for Plant Disease Detection FYP
-------------------------------------------------------------------------
This file contains:
- Hyperparameters & Image configuration
- File and directory paths
- Disease Information Encyclopedia (descriptions, symptoms, causes, remedies, prevention)
"""

import os

# Base Directory of the Project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory Paths
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
VAL_DIR = os.path.join(DATASET_DIR, 'validation')
TEST_DIR = os.path.join(DATASET_DIR, 'test')


MODEL_DIR = BASE_DIR

MODEL_PATH = os.path.join(
    BASE_DIR,
    'plant_disease_model.keras'
)

CLASS_INDICES_PATH = os.path.join(
    BASE_DIR,
    'class_indices.json'
)

HISTORY_PATH = os.path.join(
    BASE_DIR,
    'training_history.json'
)

STATIC_DIR = os.path.join(BASE_DIR, 'static')
UPLOAD_FOLDER = os.path.join(STATIC_DIR, 'uploads')
CHARTS_DIR = os.path.join(STATIC_DIR, 'charts')

# Allowed extensions for uploaded leaf images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload size

# Model Hyperparameters
# 128x128 provides high accuracy with fast training speeds on typical student laptops (CPU/GPU)
IMAGE_SIZE = (128, 128)
IMAGE_SHAPE = (128, 128, 3)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
CONFIDENCE_THRESHOLD = 0.60  # Minimum confidence (60%) for reliable detection

# Default supported classes for initial FYP scope (can easily be extended)
DEFAULT_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Grape___Black_rot",
    "Grape___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy"
]

# Comprehensive Disease Knowledge Base
# Contains practical agriculture & plant pathology advice for farmers/students
DISEASE_INFO = {
    "Apple___Apple_scab": {
        "plant": "Apple",
        "disease_name": "Apple Scab",
        "status": "Diseased",
        "severity": "Moderate to High",
        "pathogen": "Venturia inaequalis (Fungus)",
        "description": "Apple scab is a serious fungal disease that attacks both leaves and fruit, causing dark, olive-green to black scabby lesions.",
        "symptoms": [
            "Olive-green velvety spots on leaf upper surfaces",
            "Yellowing and premature drop of affected leaves",
            "Dark, corky, scabby lesions on fruit",
            "Distorted or cracked fruit development"
        ],
        "cause": "Fungal spores overwinter in fallen dead leaves and infect fresh foliage during wet, cool spring weather.",
        "recommendation": "Apply protective fungicide sprays (e.g., Captan, Mancozeb, or copper-based sprays) at green tip stage. Rake and destroy fallen leaves in autumn.",
        "prevention": [
            "Plant scab-resistant apple cultivars (e.g., Liberty, Enterprise, Freedom).",
            "Prune trees annually to enhance air circulation and quick leaf drying.",
            "Avoid overhead irrigation; water at soil level."
        ]
    },
    "Apple___Black_rot": {
        "plant": "Apple",
        "disease_name": "Black Rot",
        "status": "Diseased",
        "severity": "High",
        "pathogen": "Botryosphaeria obtusa (Fungus)",
        "description": "Black rot causes leaf spotting (frog-eye leaf spot), cankers on branches, and brown to black rot on developing fruit.",
        "symptoms": [
            "Small purple spots on leaves expanding with tan centers ('frog-eye' appearance)",
            "Firm brown rot on fruit beginning near calyx, eventually turning coal black",
            "Sunken reddish-brown cankers on bark"
        ],
        "cause": "Fungus survives in mummified fruit, dead wood, and bark cankers.",
        "recommendation": "Prune out diseased cankers, dead branches, and remove all mummified fruit from trees and ground.",
        "prevention": [
            "Maintain tree vigor through balanced fertilization.",
            "Apply fungicides such as Captan or Thiophanate-methyl starting at petal fall.",
            "Disinfect pruning shears with 70% alcohol between cuts."
        ]
    },
    "Apple___Cedar_apple_rust": {
        "plant": "Apple",
        "disease_name": "Cedar Apple Rust",
        "status": "Diseased",
        "severity": "Moderate",
        "pathogen": "Gymnosporangium juniperi-virginianae (Fungus)",
        "description": "A fungal disease requiring both apple trees and Eastern red cedar (juniper) trees to complete its life cycle.",
        "symptoms": [
            "Bright orange-yellow circular spots on upper leaf surfaces",
            "Small tube-like fungal projections (aecia) on leaf undersides",
            "Premature defoliation in severe infections"
        ],
        "cause": "Windborne basidiospores from galls on nearby juniper/cedar trees during warm spring rains.",
        "recommendation": "Apply fungicides like Myclobutanil or Immunox when orange gelatinous galls appear on nearby cedars.",
        "prevention": [
            "Remove Eastern red cedar trees within a 1-mile radius if practical.",
            "Choose rust-resistant varieties such as Redfree, Priscilla, or Enterprise.",
            "Regularly inspect foliage during spring."
        ]
    },
    "Apple___healthy": {
        "plant": "Apple",
        "disease_name": "None (Healthy Plant)",
        "status": "Healthy",
        "severity": "None",
        "pathogen": "None",
        "description": "The apple leaf exhibits strong vigor, uniform green pigmentation, clear vein structures, and no evidence of fungal or bacterial pathogens.",
        "symptoms": [
            "Uniform rich green coloration",
            "No necrotic spots or chlorosis",
            "Smooth and intact leaf margin"
        ],
        "cause": "Optimal soil nutrients, balanced hydration, and good orchard management.",
        "recommendation": "Continue regular orchard monitoring, proper watering schedules, and seasonal maintenance.",
        "prevention": [
            "Maintain annual dormant pruning.",
            "Perform soil testing every 2-3 years for optimal pH (6.0 - 6.5).",
            "Apply organic mulch around tree bases."
        ]
    },
    "Corn___Common_rust": {
        "plant": "Corn (Maize)",
        "disease_name": "Common Rust",
        "status": "Diseased",
        "severity": "Moderate",
        "pathogen": "Puccinia sorghi (Fungus)",
        "description": "Common rust produces powdery brick-red to brown pustules over both leaf surfaces of maize plants.",
        "symptoms": [
            "Oval to elongate cinnamon-brown pustules on both leaf surfaces",
            "Pustules rupture epidermal tissue exposing powdery spores",
            "Leaves turn yellow, dry out, and die prematurely under heavy infestation"
        ],
        "cause": "Airborne spores carried on prevailing winds from warmer southern regions under high humidity and cool to moderate temperatures.",
        "recommendation": "Apply foliar fungicides (triazoles or strobilurins) if rust appears early before tassel stage on susceptible hybrids.",
        "prevention": [
            "Plant resistant corn hybrids with Rp gene resistance.",
            "Ensure balanced nitrogen and potassium fertilization.",
            "Plant early in the season to evade late-season spore showers."
        ]
    },
    "Corn___Northern_Leaf_Blight": {
        "plant": "Corn (Maize)",
        "disease_name": "Northern Leaf Blight",
        "status": "Diseased",
        "severity": "High",
        "pathogen": "Exserohilum turcicum (Fungus)",
        "description": "A destructive foliar fungal disease causing large cigar-shaped necrotic lesions on corn leaves, drastically reducing photosynthetic area.",
        "symptoms": [
            "Long, elliptical, grayish-green or tan lesions (1 to 6 inches long)",
            "Lesions develop dark fungal spore mats in damp weather",
            "Extensive leaf blighting starting from lower canopy moving upwards"
        ],
        "cause": "Fungus overwinters in crop residue and spreads during moderate temperatures (18-27°C) accompanied by heavy dews.",
        "recommendation": "Apply fungicides (e.g., Pyraclostrobin, Azoxystrobin) at first sign of disease near tasseling.",
        "prevention": [
            "Rotate corn with non-host crops like soybeans for 1-2 seasons.",
            "Incorporate and till crop debris to speed decomposition.",
            "Select hybrids with high multi-gene resistance ratings."
        ]
    },
    "Corn___healthy": {
        "plant": "Corn (Maize)",
        "disease_name": "None (Healthy Plant)",
        "status": "Healthy",
        "severity": "None",
        "pathogen": "None",
        "description": "The corn leaf displays robust chlorophyll content, clean venation, and zero signs of blighting or rust lesions.",
        "symptoms": [
            "Vibrant, dark green foliage",
            "Intact leaf edges and central midrib",
            "No pustules, stripes, or necrotic margins"
        ],
        "cause": "Adequate soil nitrogen, balanced moisture, and effective crop scouting.",
        "recommendation": "Maintain standard agronomic practices and timely irrigation during critical growth stages.",
        "prevention": [
            "Follow recommended seed spacing and depth.",
            "Practice continuous weed control to eliminate alternate hosts.",
            "Monitor soil fertility."
        ]
    },
    "Grape___Black_rot": {
        "plant": "Grape",
        "disease_name": "Black Rot",
        "status": "Diseased",
        "severity": "High",
        "pathogen": "Guignardia bidwellii (Fungus)",
        "description": "A severe fungal disease that can destroy 80-100% of grape crop yield if unmanaged in warm, humid climates.",
        "symptoms": [
            "Small circular reddish-brown leaf spots with dark borders",
            "Tiny black specks (pycnidia) arranged in rings inside leaf spots",
            "Berries shrivel into hard, black, wrinkled mummies"
        ],
        "cause": "Overwintering in mummified berries on vines or soil; splash-dispersed by spring rainfall.",
        "recommendation": "Spray fungicides (Mancozeb, Ziram, or Myclobutanil) starting at bud break until 4-5 weeks post-bloom.",
        "prevention": [
            "Sanitation: remove and destroy all mummified grapes during winter pruning.",
            "Canopy management: trellis training to maximize sunlight penetration and air flow.",
            "Avoid overhead sprinkling."
        ]
    },
    "Grape___healthy": {
        "plant": "Grape",
        "disease_name": "None (Healthy Plant)",
        "status": "Healthy",
        "severity": "None",
        "pathogen": "None",
        "description": "The grapevine leaf is completely healthy with vibrant color, prominent palmate venation, and no fungal patches.",
        "symptoms": [
            "Lush green surface without discoloration",
            "Uniform leaf lobe structure",
            "No mildew, spots, or shriveling"
        ],
        "cause": "Good canopy aeration, optimal sunlight, and disease prevention regime.",
        "recommendation": "Keep up regular canopy training and shoot positioning for continued vitality.",
        "prevention": [
            "Keep grass and weeds mowed short beneath vines.",
            "Prune dormant vines cleanly each winter.",
            "Monitor leaf undersides during humid weather."
        ]
    },
    "Potato___Early_blight": {
        "plant": "Potato",
        "disease_name": "Early Blight",
        "status": "Diseased",
        "severity": "Moderate to High",
        "pathogen": "Alternaria solani (Fungus)",
        "description": "Early blight is a very common fungal disease producing distinctive concentric ring spots on older potato leaves.",
        "symptoms": [
            "Dark brown to black spots with concentric rings ('target-board' pattern)",
            "Yellow halo surrounding leaf lesions",
            "Lower, older leaves yellow and drop first"
        ],
        "cause": "High humidity, alternating wet and dry conditions, and stress on older plant tissue.",
        "recommendation": "Apply protectant fungicides like Chlorothalonil or copper fungicides when symptoms first appear.",
        "prevention": [
            "Practice a 3-4 year crop rotation away from solanaceous plants (tomatoes, peppers, potatoes).",
            "Ensure uniform soil moisture and adequate nitrogen levels.",
            "Destroy crop residues after harvest."
        ]
    },
    "Potato___Late_blight": {
        "plant": "Potato",
        "disease_name": "Late Blight",
        "status": "Diseased",
        "severity": "Very High / Critical",
        "pathogen": "Phytophthora infestans (Oomycete)",
        "description": "Late blight is the devastating pathogen responsible for the Irish Potato Famine. It can destroy entire fields within days under cool, wet conditions.",
        "symptoms": [
            "Water-soaked dark brown to purplish lesions on leaves and stems",
            "White fungal-like fuzzy growth on underside of leaves in humid mornings",
            "Rapid collapse and rotting of plant foliage and tuber decay"
        ],
        "cause": "Rapidly spreading windborne sporangia under cool (15-20°C), wet, foggy conditions.",
        "recommendation": "Immediate application of systemic fungicides (e.g., Metalaxyl, Dimethomorph, Cymoxanil) combined with protectants.",
        "prevention": [
            "Plant only certified disease-free seed tubers.",
            "Eliminate cull piles and volunteer potato plants.",
            "Ensure wide row spacing for maximum leaf drying."
        ]
    },
    "Potato___healthy": {
        "plant": "Potato",
        "disease_name": "None (Healthy Plant)",
        "status": "Healthy",
        "severity": "None",
        "pathogen": "None",
        "description": "The potato foliage shows strong vigor, healthy green coloration, and no signs of blight lesions or chlorosis.",
        "symptoms": [
            "Even dark green leaf color",
            "Sturdy leaf stems and petioles",
            "Clean leaf surface without spots or mold"
        ],
        "cause": "Clean seed stock, good drainage, and balanced N-P-K nutrient availability.",
        "recommendation": "Maintain hill cultivation and regular inspection for early signs of disease.",
        "prevention": [
            "Ensure well-drained soil with pH 5.5 - 6.5.",
            "Keep soil hilled around developing tubers.",
            "Avoid overhead watering late in the afternoon."
        ]
    },
    "Tomato___Bacterial_spot": {
        "plant": "Tomato",
        "disease_name": "Bacterial Spot",
        "status": "Diseased",
        "severity": "High",
        "pathogen": "Xanthomonas species (Bacteria)",
        "description": "Bacterial spot is a widespread bacterial disease causing small, dark, water-soaked spots on tomato leaves, stems, and fruits.",
        "symptoms": [
            "Small (1/8 inch), angular, dark brown to black greasy spots",
            "Yellow halos surrounding older spots",
            "Extensive leaf defoliation exposing green fruit to sunscald",
            "Black scabby spots on fruit surface"
        ],
        "cause": "Bacterial spread by splashing rain, wind-driven water, and handling wet plants.",
        "recommendation": "Apply fixed copper bactericides mixed with Mancozeb spray at early symptom onset.",
        "prevention": [
            "Use certified pathogen-free treated seeds and healthy transplants.",
            "Avoid overhead sprinkler irrigation.",
            "Do not work in fields when tomato foliage is wet."
        ]
    },
    "Tomato___Early_blight": {
        "plant": "Tomato",
        "disease_name": "Early Blight",
        "status": "Diseased",
        "severity": "Moderate to High",
        "pathogen": "Alternaria linariae / Alternaria solani (Fungus)",
        "description": "A very common fungal issue characterized by target-like ring patterns on lower tomato foliage.",
        "symptoms": [
            "Dark brown concentric circular spots ('target board') on older leaves",
            "Leaves turn chlorotic (yellow) around spots and drop off",
            "Collar rot lesions on lower stems of seedlings"
        ],
        "cause": "Spreads from soil splashes containing fungal spores during warm (24-29°C), humid weather.",
        "recommendation": "Remove infected lower leaves immediately. Spray organic copper or Chlorothalonil every 7-10 days.",
        "prevention": [
            "Mulch heavily around tomato base (straw or plastic) to prevent soil splash.",
            "Prune bottom 12 inches of foliage as plants grow.",
            "Rotate crops annually with non-solanaceous plants."
        ]
    },
    "Tomato___Late_blight": {
        "plant": "Tomato",
        "disease_name": "Late Blight",
        "status": "Diseased",
        "severity": "Critical",
        "pathogen": "Phytophthora infestans (Oomycete)",
        "description": "Extremely destructive disease that turns healthy tomato plants into brown, rotted foliage in just a few days.",
        "symptoms": [
            "Large, irregularly shaped, water-soaked oily grayish-green lesions",
            "Delicate white fungal-like fuzz on undersides of leaves during high moisture",
            "Stems develop dark brown greasy girdling lesions",
            "Firm brown greasy decay on tomato fruit"
        ],
        "cause": "Airborne spores carried over long distances during cool, rainy, overcast weather.",
        "recommendation": "Immediately rogue and bag severely infected plants. Apply targeted fungicides (e.g., Chlorothalonil, Copper, Mandipropamid).",
        "prevention": [
            "Choose resistant varieties (e.g., Mountain Magic, Defiant PhR, Plum Regal).",
            "Provide ample spacing between plants for airflow.",
            "Destroy volunteer tomato and potato plants."
        ]
    },
    "Tomato___healthy": {
        "plant": "Tomato",
        "disease_name": "None (Healthy Plant)",
        "status": "Healthy",
        "severity": "None",
        "pathogen": "None",
        "description": "The tomato leaf exhibits pristine physiological health with rich green color, crisp edges, and no lesions.",
        "symptoms": [
            "Vibrant green compound leaves",
            "Clean stems and leaflet margins",
            "No spots, wilting, or yellowing"
        ],
        "cause": "Balanced fertigation, appropriate staking, and clean soil conditions.",
        "recommendation": "Continue standard stake/cage support, routine watering at base, and balanced fertilization.",
        "prevention": [
            "Keep soil mulch in place.",
            "Water early in the morning using drip or soaker hoses.",
            "Check leaves weekly for any early pest or disease signs."
        ]
    }
}

def get_disease_details(class_name: str) -> dict:
    """Helper function to safely retrieve disease information dictionary."""
    if class_name in DISEASE_INFO:
        return DISEASE_INFO[class_name]
    
    # Fallback if an unknown or newly added class is detected
    clean_name = class_name.replace("___", " - ").replace("_", " ")
    is_healthy = "healthy" in class_name.lower()
    return {
        "plant": class_name.split("___")[0].replace("_", " "),
        "disease_name": "Healthy Leaf" if is_healthy else clean_name,
        "status": "Healthy" if is_healthy else "Diseased",
        "severity": "None" if is_healthy else "Unknown",
        "pathogen": "None" if is_healthy else "Pathogen under evaluation",
        "description": f"Analysis report for {clean_name}.",
        "symptoms": ["Detailed symptoms not indexed in local knowledge base."],
        "cause": "Under investigation.",
        "recommendation": "Consult a local agricultural extension officer for specific treatment advice.",
        "prevention": ["Maintain good cultural practices, sanitation, and balanced watering."]
    }

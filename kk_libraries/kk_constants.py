#!/usr/bin/env python
# _*_ encoding: utf-8 _*_
# @Author kkutysllb


text_labels_fashion_mnist = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag',
                             'ankle boot']
text_labels_mnist = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
text_labels_cifar10 = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
text_cat_dog = ['cat', 'dog']
text_fruits = ['apple', 'banana', 'grape', 'orange', 'pear']
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
text_labels_imagenet = [
    'tench', 'goldfish', 'great_white_shark', 'tiger_shark', 'hammerhead_shark', 'electric_ray', 'stingray', 'cock', 'hen', 'ostrich',
    'brambling', 'goldfinch', 'house_finch', 'junco', 'indigo_bunting', 'robin', 'bulbul', 'jay', 'magpie', 'chickadee',
    'water_ouzel', 'kite', 'bald_eagle', 'vulture', 'great_grey_owl', 'european_fire_salamander', 'common_newt', 'eft', 'spotted_salamander', 'axolotl',
    'bullfrog', 'tree_frog', 'tailed_frog', 'loggerhead', 'leatherback_turtle', 'mud_turtle', 'terrapin', 'box_turtle', 'banded_gecko', 'common_iguana',
    'american_chameleon', 'whiptail', 'agama', 'frilled_lizard', 'alligator_lizard', 'gila_monster', 'green_lizard', 'african_chameleon', 'komodo_dragon', 'african_crocodile',
    'american_alligator', 'triceratops', 'thunder_snake', 'ringneck_snake', 'hognose_snake', 'green_snake', 'king_snake', 'garter_snake', 'water_snake', 'vine_snake',
    'night_snake', 'boa_constrictor', 'rock_python', 'indian_cobra', 'green_mamba', 'sea_snake', 'horned_viper', 'diamondback', 'sidewinder', 'trilobite',
    'harvestman', 'scorpion', 'black_and_gold_garden_spider', 'barn_spider', 'garden_spider', 'black_widow', 'tarantula', 'wolf_spider', 'tick', 'centipede',
    'black_grouse', 'ptarmigan', 'ruffed_grouse', 'prairie_chicken', 'peacock', 'quail', 'partridge', 'african_grey', 'macaw', 'sulphur_crested_cockatoo',
    'lorikeet', 'coucal', 'bee_eater', 'hornbill', 'hummingbird', 'jacamar', 'toucan', 'drake', 'red_breasted_merganser', 'goose',
    'black_swan', 'tusker', 'echidna', 'platypus', 'wallaby', 'koala', 'wombat', 'jellyfish', 'sea_anemone', 'brain_coral',
    'flatworm', 'nematode', 'conch', 'snail', 'slug', 'sea_slug', 'chiton', 'chambered_nautilus', 'dungeness_crab', 'rock_crab',
    'fiddler_crab', 'king_crab', 'american_lobster', 'spiny_lobster', 'crayfish', 'hermit_crab', 'isopod', 'white_stork', 'black_stork', 'spoonbill',
    'flamingo', 'little_blue_heron', 'american_egret', 'bittern', 'crane', 'limpkin', 'european_gallinule', 'american_coot', 'bustard', 'ruddy_turnstone',
    'red_backed_sandpiper', 'redshank', 'dowitcher', 'oystercatcher', 'pelican', 'king_penguin', 'albatross', 'grey_whale', 'killer_whale', 'dugong',
    'sea_lion', 'chihuahua', 'japanese_spaniel', 'maltese_dog', 'pekinese', 'shih_tzu', 'blenheim_spaniel', 'papillon', 'toy_terrier', 'rhodesian_ridgeback',
    'afghan_hound', 'basset', 'beagle', 'bloodhound', 'bluetick', 'black_and_tan_coonhound', 'walker_hound', 'english_foxhound', 'redbone', 'borzoi',
    'irish_wolfhound', 'italian_greyhound', 'whippet', 'ibizan_hound', 'norwegian_elkhound', 'otterhound', 'saluki', 'scottish_deerhound', 'weimaraner', 'staffordshire_bullterrier',
    'american_staffordshire_terrier', 'bedlington_terrier', 'border_terrier', 'kerry_blue_terrier', 'irish_terrier', 'norfolk_terrier', 'norwich_terrier', 'yorkshire_terrier', 'wire_haired_fox_terrier', 'lakeland_terrier',
    'sealyham_terrier', 'airedale', 'cairn', 'australian_terrier', 'dandie_dinmont', 'boston_bull', 'miniature_schnauzer', 'giant_schnauzer', 'standard_schnauzer', 'scotch_terrier',
    'tibetan_terrier', 'silky_terrier', 'soft_coated_wheaten_terrier', 'west_highland_white_terrier', 'lhasa', 'flat_coated_retriever', 'curly_coated_retriever', 'golden_retriever', 'labrador_retriever', 'chesapeake_bay_retriever',
    'german_short_haired_pointer', 'vizsla', 'english_setter', 'irish_setter', 'gordon_setter', 'brittany_spaniel', 'clumber', 'english_springer', 'welsh_springer_spaniel', 'cocker_spaniel',
    'sussex_spaniel', 'irish_water_spaniel', 'kuvasz', 'schipperke', 'groenendael', 'malinois', 'briard', 'kelpie', 'komondor', 'old_english_sheepdog',
    'shetland_sheepdog', 'collie', 'border_collie', 'bouvier_des_flandres', 'rottweiler', 'german_shepherd', 'doberman', 'miniature_pinscher', 'greater_swiss_mountain_dog', 'bernese_mountain_dog',
    'appenzeller', 'entlebucher', 'boxer', 'bull_mastiff', 'tibetan_mastiff', 'french_bulldog', 'great_dane', 'saint_bernard', 'eskimo_dog', 'malamute',
    'siberian_husky', 'dalmatian', 'affenpinscher', 'basenji', 'pug', 'leonberg', 'newfoundland', 'great_pyrenees', 'samoyed', 'pomeranian',
    'chow', 'keeshond', 'brabancon_griffon', 'pembroke', 'cardigan', 'toy_poodle', 'miniature_poodle', 'standard_poodle', 'mexican_hairless', 'timber_wolf',
    'white_wolf', 'red_wolf', 'coyote', 'dingo', 'dhole', 'african_hunting_dog', 'hyena', 'red_fox', 'kit_fox', 'arctic_fox',
    'grey_fox', 'tabby', 'tiger_cat', 'persian_cat', 'siamese_cat', 'egyptian_cat', 'cougar', 'lynx', 'leopard', 'snow_leopard',
    'jaguar', 'lion', 'tiger', 'cheetah', 'brown_bear', 'american_black_bear', 'ice_bear', 'sloth_bear', 'mongoose', 'meerkat',
    'tiger_beetle', 'ladybug', 'ground_beetle', 'long_horned_beetle', 'leaf_beetle', 'dung_beetle', 'rhinoceros_beetle', 'weevil', 'fly', 'bee',
    'ant', 'grasshopper', 'cricket', 'walking_stick', 'cockroach', 'mantis', 'cicada', 'leafhopper', 'lacewing', 'dragonfly',
    'damselfly', 'admiral', 'ringlet', 'monarch', 'cabbage_butterfly', 'sulphur_butterfly', 'lycaenid', 'starfish', 'sea_urchin', 'sea_cucumber',
    'wood_rabbit', 'hare', 'angora', 'hamster', 'porcupine', 'fox_squirrel', 'marmot', 'beaver', 'guinea_pig', 'sorrel',
    'zebra', 'hog', 'wild_boar', 'warthog', 'hippopotamus', 'ox', 'water_buffalo', 'bison', 'ram', 'bighorn',
    'ibex', 'hartebeest', 'impala', 'gazelle', 'arabian_camel', 'llama', 'weasel', 'mink', 'polecat', 'black_footed_ferret',
    'otter', 'skunk', 'badger', 'armadillo', 'three_toed_sloth', 'orangutan', 'gorilla', 'chimpanzee', 'gibbon', 'siamang',
    'guenon', 'patas', 'baboon', 'macaque', 'langur', 'colobus', 'proboscis_monkey', 'marmoset', 'capuchin', 'howler_monkey',
    'titi', 'spider_monkey', 'squirrel_monkey', 'madagascar_cat', 'indri', 'indian_elephant', 'african_elephant', 'lesser_panda', 'giant_panda', 'barracouta',
    'eel', 'coho', 'rock_beauty', 'anemone_fish', 'sturgeon', 'gar', 'lionfish', 'puffer', 'abacus', 'abaya',
    'academic_gown', 'accordion', 'acoustic_guitar', 'aircraft_carrier', 'airliner', 'airship', 'altar', 'ambulance', 'amphibian', 'analog_clock',
    'apiary', 'apron', 'ashcan', 'assault_rifle', 'backpack', 'bakery', 'balance_beam', 'balloon', 'ballpoint', 'band_aid',
    'banjo', 'bannister', 'barbell', 'barber_chair', 'barbershop', 'barn', 'barometer', 'barrel', 'barrow', 'baseball',
    'basketball', 'bassinet', 'bassoon', 'bathing_cap', 'bath_towel', 'bathtub', 'beach_wagon', 'beacon', 'beaker', 'bearskin',
    'beer_bottle', 'beer_glass', 'bell_cote', 'bib', 'bicycle_built_for_two', 'bikini', 'binder', 'binoculars', 'birdhouse', 'boathouse',
    'bobsled', 'bolo_tie', 'bonnet', 'bookcase', 'bookshop', 'bottlecap', 'bow', 'bow_tie', 'brass', 'brassiere',
    'breakwater', 'breastplate', 'broom', 'bucket', 'buckle', 'bulletproof_vest', 'bullet_train', 'butcher_shop', 'cab', 'caldron',
    'candle', 'cannon', 'canoe', 'can_opener', 'cardigan', 'car_mirror', 'carousel', 'carpenters_kit', 'carton', 'car_wheel',
    'cash_machine', 'cassette', 'cassette_player', 'castle', 'catamaran', 'cd_player', 'cello', 'cellular_telephone', 'chain', 'chainlink_fence',
    'chain_mail', 'chain_saw', 'chest', 'chiffonier', 'chime', 'china_cabinet', 'christmas_stocking', 'church', 'cinema', 'cleaver',
    'cliff_dwelling', 'cloak', 'clog', 'cocktail_shaker', 'coffee_mug', 'coffeepot', 'coil', 'combination_lock', 'computer_keyboard', 'confectionery',
    'container_ship', 'convertible', 'corkscrew', 'cornet', 'cowboy_boot', 'cowboy_hat', 'cradle', 'crane', 'crash_helmet', 'crate',
    'crib', 'crock_pot', 'croquet_ball', 'crutch', 'cuirass', 'dam', 'desk', 'desktop_computer', 'dial_telephone', 'diaper',
    'digital_clock', 'digital_watch', 'dining_table', 'dishrag', 'dishwasher', 'disk_brake', 'dock', 'dogsled', 'dome', 'doormat',
    'drilling_platform', 'drum', 'drumstick', 'dumbbell', 'dutch_oven', 'electric_fan', 'electric_guitar', 'electric_locomotive', 'entertainment_center', 'envelope',
    'espresso_maker', 'face_powder', 'feather_boa', 'file', 'fireboat', 'fire_engine', 'fire_screen', 'flagpole', 'flute', 'folding_chair',
    'football_helmet', 'forklift', 'fountain', 'fountain_pen', 'four_poster', 'freight_car', 'french_horn', 'frying_pan', 'fur_coat', 'garbage_truck',
    'gasmask', 'gas_pump', 'goblet', 'go_kart', 'golf_ball', 'golfcart', 'gondola', 'gong', 'gown', 'grand_piano',
    'greenhouse', 'grille', 'grocery_store', 'guillotine', 'hair_slide', 'hair_spray', 'half_track', 'hammer', 'hamper', 'hand_blower',
    'hand_held_computer', 'handkerchief', 'hard_disc', 'harmonica', 'harp', 'harvester', 'hatchet', 'holster', 'home_theater', 'honeycomb',
    'hook', 'hoopskirt', 'horizontal_bar', 'horse_cart', 'hourglass', 'ipod', 'iron', 'jack_o_lantern', 'jean', 'jeep',
    'jersey', 'jigsaw_puzzle', 'jinrikisha', 'joystick', 'kimono', 'knee_pad', 'knot', 'lab_coat', 'ladle', 'lampshade',
    'laptop', 'lawn_mower', 'lens_cap', 'letter_opener', 'library', 'lifeboat', 'lighter', 'limousine', 'liner', 'lipstick',
    'loafer', 'lotion', 'loudspeaker', 'loupe', 'lumbermill', 'magnetic_compass', 'mailbag', 'mailbox', 'maillot', 'maillot_tank_suit',
    'manhole_cover', 'maraca', 'marimba', 'mask', 'matchstick', 'maypole', 'maze', 'measuring_cup', 'medicine_chest', 'megalith',
    'microphone', 'microwave', 'military_uniform', 'milk_can', 'minibus', 'miniskirt', 'minivan', 'missile', 'mitten', 'mixing_bowl',
    'mobile_home', 'model_t', 'modem', 'monastery', 'monitor', 'moped', 'mortar', 'mortarboard', 'mosque', 'mosquito_net',
    'motor_scooter', 'mountain_bike', 'mountain_tent', 'mouse', 'mousetrap', 'moving_van', 'muzzle', 'nail', 'neck_brace', 'necklace',
    'nipple', 'notebook', 'obelisk', 'oboe', 'ocarina', 'odometer', 'oil_filter', 'organ', 'oscilloscope', 'overskirt',
    'oxcart', 'oxygen_mask', 'packet', 'paddle', 'paddlewheel', 'padlock', 'paintbrush', 'pajama', 'palace', 'panpipe',
    'paper_towel', 'parachute', 'parallel_bars', 'park_bench', 'parking_meter', 'passenger_car', 'patio', 'pay_phone', 'pedestal', 'pencil_box',
    'pencil_sharpener', 'perfume', 'petri_dish', 'photocopier', 'pick', 'pickelhaube', 'picket_fence', 'pickup', 'pier', 'piggy_bank',
    'pill_bottle', 'pillow', 'ping_pong_ball', 'pinwheel', 'pirate', 'pitcher', 'plane', 'planetarium', 'plastic_bag', 'plate_rack',
    'plow', 'plunger', 'polaroid_camera', 'pole', 'police_van', 'poncho', 'pool_table', 'pop_bottle', 'pot', 'potters_wheel',
    'power_drill', 'prayer_rug', 'printer', 'prison', 'projectile', 'projector', 'puck', 'punching_bag', 'purse', 'quill',
    'quilt', 'racer', 'racket', 'radiator', 'radio', 'radio_telescope', 'rain_barrel', 'recreational_vehicle', 'reel', 'reflex_camera',
    'refrigerator', 'remote_control', 'restaurant', 'revolver', 'rifle', 'rocking_chair', 'rotisserie', 'rubber_eraser', 'rugby_ball', 'rule',
    'running_shoe', 'safe', 'safety_pin', 'saltshaker', 'sandal', 'sarong', 'sax', 'scabbard', 'scale', 'school_bus',
    'schooner', 'scoreboard', 'screen', 'screw', 'screwdriver', 'seat_belt', 'sewing_machine', 'shield', 'shoe_shop', 'shoji',
    'shopping_basket', 'shopping_cart', 'shovel', 'shower_cap', 'shower_curtain', 'ski', 'ski_mask', 'sleeping_bag', 'slide_rule', 'sliding_door',
    'slot', 'snorkel', 'snowmobile', 'snowplow', 'soap_dispenser', 'soccer_ball', 'sock', 'solar_dish', 'sombrero', 'soup_bowl',
    'space_bar', 'space_heater', 'space_shuttle', 'spatula', 'speedboat', 'spider_web', 'spindle', 'sports_car', 'spotlight', 'stage',
    'steam_locomotive', 'steel_arch_bridge', 'steel_drum', 'stethoscope', 'stole', 'stone_wall', 'stopwatch', 'stove', 'strainer', 'streetcar',
    'stretcher', 'studio_couch', 'stupa', 'submarine', 'suit', 'sundial', 'sunglass', 'sunglasses', 'sunscreen', 'suspension_bridge',
    'swab', 'sweatshirt', 'swimming_trunks', 'swing', 'switch', 'syringe', 'table_lamp', 'tank', 'tape_player', 'teapot',
    'teddy', 'television', 'tennis_ball', 'thatch', 'theater_curtain', 'thimble', 'thresher', 'throne', 'tile_roof', 'toaster',
    'tobacco_shop', 'toilet_seat', 'torch', 'totem_pole', 'tow_truck', 'toyshop', 'tractor', 'trailer_truck', 'tray', 'trench_coat',
    'tricycle', 'trimaran', 'tripod', 'triumphal_arch', 'trolleybus', 'trombone', 'tub', 'turnstile', 'typewriter_keyboard', 'umbrella',
    'unicycle', 'upright', 'vacuum', 'vase', 'vault', 'velvet', 'vending_machine', 'vestment', 'viaduct', 'violin',
    'volleyball', 'waffle_iron', 'wall_clock', 'wallet', 'wardrobe', 'warplane', 'washbasin', 'washer', 'water_bottle', 'water_jug',
    'water_tower', 'whiskey_jug', 'whistle', 'wig', 'window_screen', 'window_shade', 'windsor_tie', 'wine_bottle', 'wing', 'wok',
    'wooden_spoon', 'wool', 'worm_fence', 'wreck', 'yawl', 'yurt', 'web_site', 'comic_book', 'crossword_puzzle', 'street_sign',
    'traffic_light', 'book_jacket', 'menu', 'plate', 'guacamole', 'consomme', 'hot_pot', 'trifle', 'ice_cream', 'ice_lolly',
    'french_loaf', 'bagel', 'pretzel', 'cheeseburger', 'hotdog', 'mashed_potato', 'head_cabbage', 'broccoli', 'cauliflower', 'zucchini',
    'spaghetti_squash', 'acorn_squash', 'butternut_squash', 'cucumber', 'artichoke', 'bell_pepper', 'cardoon', 'mushroom', 'granny_smith', 'strawberry',
    'orange', 'lemon', 'fig', 'pineapple', 'banana', 'jackfruit', 'custard_apple', 'pomegranate', 'hay', 'carbonara',
    'chocolate_sauce', 'dough', 'meat_loaf', 'pizza', 'potpie', 'burrito', 'red_wine', 'espresso', 'cup', 'eggnog',
    'alp', 'bubble', 'cliff', 'coral_reef', 'geyser', 'lakeside', 'promontory', 'sandbar', 'seashore', 'valley',
    'volcano', 'ballplayer', 'groom', 'scuba_diver', 'rapeseed', 'daisy', 'yellow_ladys_slipper', 'corn', 'acorn', 'hip',
    'buckeye', 'coral_fungus', 'agaric', 'gyromitra', 'stinkhorn', 'earthstar', 'hen_of_the_woods', 'bolete', 'ear', 'toilet_tissue'
]

text_labels_cifar100 = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

# CIFAR100的超类别（20个）和其对应的细分类别（每个超类包含5个细分类别）
cifar100_superclass = {
    'aquatic_mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
    'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
    'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
    'food_containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],
    'fruit_and_vegetables': ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
    'household_electrical_devices': ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
    'household_furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
    'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
    'large_carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
    'large_man-made_outdoor_things': ['bridge', 'castle', 'house', 'road', 'skyscraper'],
    'large_natural_outdoor_scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea'],
    'large_omnivores_and_herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
    'medium_mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
    'non-insect_invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
    'people': ['baby', 'boy', 'girl', 'man', 'woman'],
    'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
    'small_mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
    'trees': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
    'vehicles_1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
    'vehicles_2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']
}


# Food101 标签
text_labels_food101 = [
    'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad', 
    'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito', 'bruschetta', 'caesar_salad', 
    'cannoli', 'caprese_salad', 'carrot_cake', 'ceviche', 'cheesecake', 'cheese_plate', 
    'chicken_curry', 'chicken_quesadilla', 'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 
    'churros', 'clam_chowder', 'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 
    'cup_cakes', 'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots', 
    'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras', 'french_fries', 'french_onion_soup', 
    'french_toast', 'fried_calamari', 'fried_rice', 'frozen_yogurt', 'garlic_bread', 'gnocchi', 
    'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 
    'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna', 
    'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup', 
    'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes', 
    'panna_cotta', 'peking_duck', 'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib', 
    'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto', 'samosa', 'sashimi', 
    'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara', 
    'spring_rolls', 'steak', 'strawberry_shortcake', 'sushi', 'tacos', 'takoyaki', 'tiramisu', 
    'tuna_tartare', 'waffles'
]

# DTD (Describable Textures Dataset) 标签
text_labels_dtd = [
    'banded', 'blotchy', 'braided', 'bubbly', 'bumpy', 'chequered', 'cobwebbed', 'cracked', 
    'crosshatched', 'crystalline', 'dotted', 'fibrous', 'flecked', 'freckled', 'frilly', 
    'gauzy', 'grid', 'grooved', 'honeycombed', 'interlaced', 'knitted', 'lacelike', 'lined', 
    'marbled', 'matted', 'meshed', 'paisley', 'perforated', 'pitted', 'pleated', 'polka-dotted', 
    'porous', 'potholed', 'scaly', 'smeared', 'spiralled', 'sprinkled', 'stained', 'stratified', 
    'striped', 'studded', 'swirly', 'veined', 'waffled', 'woven', 'wrinkled', 'zigzagged'
]

# Flowers102 标签
text_labels_flowers102 = [
    'pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea', 'english marigold', 
    'tiger lily', 'moon orchid', 'bird of paradise', 'monkshood', 'globe thistle', 'snapdragon', 
    "colt's foot", 'king protea', 'spear thistle', 'yellow iris', 'globe-flower', 'purple coneflower', 
    'peruvian lily', 'balloon flower', 'giant white arum lily', 'fire lily', 'pincushion flower', 
    'fritillary', 'red ginger', 'grape hyacinth', 'corn poppy', 'prince of wales feathers', 
    'stemless gentian', 'artichoke', 'sweet william', 'carnation', 'garden phlox', 'love in the mist', 
    'mexican aster', 'alpine sea holly', 'ruby-lipped cattleya', 'cape flower', 'great masterwort', 
    'siam tulip', 'lenten rose', 'barbeton daisy', 'daffodil', 'sword lily', 'poinsettia', 
    'bolero deep blue', 'wallflower', 'marigold', 'buttercup', 'oxeye daisy', 'common dandelion', 
    'petunia', 'wild pansy', 'primula', 'sunflower', 'pelargonium', 'bishop of llandaff', 
    'gaura', 'geranium', 'orange dahlia', 'pink-yellow dahlia?', 'cautleya spicata', 'japanese anemone', 
    'black-eyed susan', 'silverbush', 'californian poppy', 'osteospermum', 'spring crocus', 
    'bearded iris', 'windflower', 'tree poppy', 'gazania', 'azalea', 'water lily', 'rose', 
    'thorn apple', 'morning glory', 'passion flower', 'lotus', 'toad lily', 'anthurium', 
    'frangipani', 'clematis', 'hibiscus', 'columbine', 'desert-rose', 'tree mallow', 
    'magnolia', 'cyclamen', 'watercress', 'canna lily', 'hippeastrum', 'bee balm', 'ball moss', 
    'foxglove', 'bougainvillea', 'camellia', 'mallow', 'mexican petunia', 'bromelia', 'blanket flower', 
    'trumpet creeper', 'blackberry lily'
]

text_labels_stanford_cars = [
    'AM General Hummer SUV 2000',
    'Acura Integra Type R 2001',
    'Acura RL Sedan 2012',
    'Acura TL Sedan 2012',
    'Acura TL Type-S 2008',
    'Acura TSX Sedan 2012',
    'Acura ZDX Hatchback 2012',
    'Aston Martin V8 Vantage Convertible 2012',
    'Aston Martin V8 Vantage Coupe 2012',
    'Aston Martin Virage Convertible 2012',
    'Aston Martin Virage Coupe 2012',
    'Audi A5 Coupe 2012',
    'Audi R8 Coupe 2012',
    'Audi RS 4 Convertible 2008',
    'Audi S4 Sedan 2007',
    'Audi S4 Sedan 2012',
    'Audi S5 Convertible 2012',
    'Audi S5 Coupe 2012',
    'Audi S6 Sedan 2011',
    'Audi TT Hatchback 2011',
    'Audi TT RS Coupe 2012',
    'Audi TTS Coupe 2012',
    'Audi V8 Sedan 1994',
    'BMW 1 Series Convertible 2012',
    'BMW 1 Series Coupe 2012',
    'BMW 3 Series Sedan 2012',
    'BMW 3 Series Wagon 2012',
    'BMW 6 Series Convertible 2007',
    'BMW ActiveHybrid 5 Sedan 2012',
    'BMW M3 Coupe 2012',
    'BMW M5 Sedan 2010',
    'BMW M6 Convertible 2010',
    'BMW X3 SUV 2012',
    'BMW X5 SUV 2007',
    'BMW X6 SUV 2012',
    'BMW Z4 Convertible 2012',
    'Bentley Arnage Sedan 2009',
    'Bentley Continental Flying Spur Sedan 2007',
    'Bentley Continental GT Coupe 2007',
    'Bentley Continental GT Coupe 2012',
    'Bentley Continental Supersports Conv. Convertible 2012',
    'Bentley Mulsanne Sedan 2011',
    'Bugatti Veyron 16.4 Convertible 2009',
    'Bugatti Veyron 16.4 Coupe 2009',
    'Buick Enclave SUV 2012',
    'Buick Rainier SUV 2007',
    'Buick Regal GS 2012',
    'Buick Verano Sedan 2012',
    'Cadillac CTS-V Sedan 2012',
    'Cadillac Escalade EXT Crew Cab 2007',
    'Cadillac SRX SUV 2012',
    'Chevrolet Avalanche Crew Cab 2012',
    'Chevrolet Camaro Convertible 2012',
    'Chevrolet Cobalt SS 2010',
    'Chevrolet Corvette Convertible 2012',
    'Chevrolet Corvette Ron Fellows Edition Z06 2007',
    'Chevrolet Corvette ZR1 2012',
    'Chevrolet Express Cargo Van 2007',
    'Chevrolet Express Van 2007',
    'Chevrolet HHR SS 2010',
    'Chevrolet Impala Sedan 2007',
    'Chevrolet Malibu Hybrid Sedan 2010',
    'Chevrolet Malibu Sedan 2007',
    'Chevrolet Monte Carlo Coupe 2007',
    'Chevrolet Silverado 1500 Classic Extended Cab 2007',
    'Chevrolet Silverado 1500 Extended Cab 2012',
    'Chevrolet Silverado 1500 Hybrid Crew Cab 2012',
    'Chevrolet Silverado 1500 Regular Cab 2012',
    'Chevrolet Silverado 2500HD Regular Cab 2012',
    'Chevrolet Sonic Sedan 2012',
    'Chevrolet Tahoe Hybrid SUV 2012',
    'Chevrolet TrailBlazer SS 2009',
    'Chevrolet Traverse SUV 2012',
    'Chrysler 300 SRT-8 2010',
    'Chrysler Aspen SUV 2009',
    'Chrysler Crossfire Convertible 2008',
    'Chrysler PT Cruiser Convertible 2008',
    'Chrysler Sebring Convertible 2010',
    'Chrysler Town and Country Minivan 2012',
    'Daewoo Nubira Wagon 2002',
    'Dodge Caliber Wagon 2007',
    'Dodge Caliber Wagon 2012',
    'Dodge Caravan Minivan 1997',
    'Dodge Challenger SRT8 2011',
    'Dodge Charger SRT-8 2009',
    'Dodge Charger Sedan 2012',
    'Dodge Dakota Club Cab 2007',
    'Dodge Dakota Crew Cab 2010',
    'Dodge Durango SUV 2007',
    'Dodge Durango SUV 2012',
    'Dodge Journey SUV 2012',
    'Dodge Magnum Wagon 2008',
    'Dodge Ram Pickup 3500 Crew Cab 2010',
    'Dodge Ram Pickup 3500 Quad Cab 2009',
    'Dodge Sprinter Cargo Van 2009',
    'Eagle Talon Hatchback 1998',
    'FIAT 500 Abarth 2012',
    'FIAT 500 Convertible 2012',
    'Ferrari 458 Italia Convertible 2012',
    'Ferrari 458 Italia Coupe 2012',
    'Ferrari California Convertible 2012',
    'Ferrari FF Coupe 2012',
    'Fisker Karma Sedan 2012',
    'Ford E-Series Wagon Van 2012',
    'Ford Edge SUV 2012',
    'Ford Expedition EL SUV 2009',
    'Ford F-150 Regular Cab 2007',
    'Ford F-150 Regular Cab 2012',
    'Ford F-450 Super Duty Crew Cab 2012',
    'Ford Fiesta Sedan 2012',
    'Ford Focus Sedan 2007',
    'Ford Freestar Minivan 2007',
    'Ford GT Coupe 2006',
    'Ford Mustang Convertible 2007',
    'Ford Ranger SuperCab 2011',
    'GMC Acadia SUV 2012',
    'GMC Canyon Extended Cab 2012',
    'GMC Savana Van 2012',
    'GMC Terrain SUV 2012',
    'GMC Yukon Hybrid SUV 2012',
    'Geo Metro Convertible 1993',
    'HUMMER H2 SUT Crew Cab 2009',
    'HUMMER H3T Crew Cab 2010',
    'Honda Accord Coupe 2012',
    'Honda Accord Sedan 2012',
    'Honda Odyssey Minivan 2007',
    'Honda Odyssey Minivan 2012',
    'Hyundai Accent Sedan 2012',
    'Hyundai Azera Sedan 2012',
    'Hyundai Elantra Sedan 2007',
    'Hyundai Elantra Touring Hatchback 2012',
    'Hyundai Genesis Sedan 2012',
    'Hyundai Santa Fe SUV 2012',
    'Hyundai Sonata Hybrid Sedan 2012',
    'Hyundai Sonata Sedan 2012',
    'Hyundai Tucson SUV 2012',
    'Hyundai Veloster Hatchback 2012',
    'Hyundai Veracruz SUV 2012',
    'Infiniti G Coupe IPL 2012',
    'Infiniti QX56 SUV 2011',
    'Isuzu Ascender SUV 2008',
    'Jaguar XK XKR 2012',
    'Jeep Compass SUV 2012',
    'Jeep Grand Cherokee SUV 2012',
    'Jeep Liberty SUV 2012',
    'Jeep Patriot SUV 2012',
    'Jeep Wrangler SUV 2012',
    'Lamborghini Aventador Coupe 2012',
    'Lamborghini Diablo Coupe 2001',
    'Lamborghini Gallardo LP 570-4 Superleggera 2012',
    'Lamborghini Reventon Coupe 2008',
    'Land Rover LR2 SUV 2012',
    'Land Rover Range Rover SUV 2012',
    'Lincoln Town Car Sedan 2011',
    'MINI Cooper Roadster Convertible 2012',
    'Maybach Landaulet Convertible 2012',
    'Mazda Tribute SUV 2011',
    'McLaren MP4-12C Coupe 2012',
    'Mercedes-Benz 300-Class Convertible 1993',
    'Mercedes-Benz C-Class Sedan 2012',
    'Mercedes-Benz E-Class Sedan 2012',
    'Mercedes-Benz S-Class Sedan 2012',
    'Mercedes-Benz SL-Class Coupe 2009',
    'Mercedes-Benz Sprinter Van 2012',
    'Mitsubishi Lancer Sedan 2012',
    'Nissan 240SX Coupe 1998',
    'Nissan Juke Hatchback 2012',
    'Nissan Leaf Hatchback 2012',
    'Nissan NV Passenger Van 2012',
    'Plymouth Neon Coupe 1999',
    'Porsche Panamera Sedan 2012',
    'Ram C/V Cargo Van Minivan 2012',
    'Rolls-Royce Ghost Sedan 2012',
    'Rolls-Royce Phantom Drophead Coupe Convertible 2012',
    'Rolls-Royce Phantom Sedan 2012',
    'Scion xD Hatchback 2012',
    'Spyker C8 Convertible 2009',
    'Spyker C8 Coupe 2009',
    'Suzuki Aerio Sedan 2007',
    'Suzuki Kizashi Sedan 2012',
    'Suzuki SX4 Hatchback 2012',
    'Suzuki SX4 Sedan 2012',
    'Tesla Model S Sedan 2012',
    'Toyota 4Runner SUV 2012',
    'Toyota Camry Sedan 2012',
    'Toyota Corolla Sedan 2012',
    'Toyota FJ Cruiser SUV 2012',
    'Toyota Sequoia SUV 2012',
    'Toyota Tacoma Extended Cab 2012',
    'Toyota Yaris Hatchback 2012',
    'Volkswagen Beetle Hatchback 2012',
    'Volkswagen Golf Hatchback 1991',
    'Volkswagen Golf Hatchback 2012',
    'Volvo 240 Sedan 1993',
    'Volvo C30 Hatchback 2012',
    'Volvo XC90 SUV 2007',
    'smart fortwo Convertible 2012'
]


text_labels_frutis100 = ['abiu', 'acai', 'acerola', 'ackee', 'ambarella', 'apple', 'apricot', 'avocado', 'banana', 
                         'barbadine', 'barberry', 'betel_nut', 'bitter_gourd', 'black_berry', 'black_mullberry', 
                         'brazil_nut', 'camu_camu', 'cashew', 'cempedak', 'chenet', 'cherimoya', 'chico', 'chokeberry',
                         'cluster_fig', 'coconut', 'corn_kernel', 'cranberry', 'cupuacu', 'custard_apple', 'damson', 
                         'dewberry', 'dragonfruit', 'durian', 'eggplant', 'elderberry', 'emblic', 'feijoa', 'fig',
                         'finger_lime', 'gooseberry', 'goumi', 'grape', 'grapefruit', 'greengage', 'grenadilla', 
                         'guava', 'hard_kiwi', 'hawthorn', 'hog_plum', 'horned_melon', 'indian_strawberry', 'jaboticaba', 
                         'jackfruit', 'jalapeno', 'jamaica_cherry', 'jambul', 'jocote', 'jujube', 'kaffir_lime', 'kumquat', 
                         'lablab', 'langsat', 'longan', 'mabolo', 'malay_apple', 'mandarine', 'mango', 'mangosteen',
                         'medlar', 'mock_strawberry', 'morinda', 'mountain_soursop', 'oil_palm', 'olive', 'otaheite_apple', 
                         'papaya', 'passion_fruit', 'pawpaw', 'pea', 'pineapple', 'plumcot', 'pomegranate', 'prikly_pear', 'quince',
                         'rambutan', 'raspberry', 'redcurrant', 'rose_hip', 'rose_leaf_bramble', 'salak', 'santol', 'sapodilla',
                         'sea_buckthorn', 'strawberry_guava', 'sugar_apple', 'taxus_baccata', 'ugli_fruit', 'white_currant', 
                         'yali_pear', 'yellow_plum']
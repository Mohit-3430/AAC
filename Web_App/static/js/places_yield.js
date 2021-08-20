var state_yield = new Array('Andaman and Nicobar Islands', 'Arunachal Pradesh', 'Bihar','Chhattisgarh', 'Himachal Pradesh', 'Jammu and Kashmir','Jharkhand', 'Kerala', 'Odisha', 'Punjab', 'Tamil Nadu','Uttarakhand')


var crop_yield = new Array();
crop_yield[0] = ""
crop_yield[1] = "Arhar/Tur,Banana,Dry chillies,Dry ginger,Groundnut,Maize,Moong(Green Gram),Rice,Sugarcane,Sunflower,Sweet potato,Turmeric,Urad "
crop_yield[2] = "Dry chillies,Dry ginger,Groundnut,Maize,Potato,Rapeseed &Mustard,Rice,Sesamum,Small millets,Sugarcane,Sunflower,Turmeric,Wheat"
crop_yield[3] = "Arhar/Tur,Bajra,Banana,Barley,Castor seed,Coriander,Cotton(lint),Dry chillies,Dry ginger,Garlic,Gram,Groundnut,Horse-gram,Jowar,Khesari,Linseed,Maize,Masoor,Moong(Gree Gram),Onion,Peas & beans (Pulses),Potato,Ragi,Rapeseed &Mustard,Rice,Sesamum,Small millets,Sugarcane,Sunflower,Sweet potato,Turmeric,Urad,Wheat"
crop_yield[4] = "Arhar/Tur,Bajra,Banana,Barley,Castorseed,Coriander,Cotton(lint),Dry chillies,Dry ginger,garlic,Gram,Groundnut,Horse-gram,Jowar,Linseed,Maize,Masoor,Moong(Green Gram),Onion,Peas & beans (Pulses),Potato,Ragi,Rapeseed &Mustard,Rice,Sesamum,Small millets,Sugarcane,Sunflower,Sweet potato,Turmeric,Urad,Wheat"
crop_yield[5] = "Arhar/Tur,Bajra,Coriander,Cotton(lint),Dry chillies,Dry ginger,Garlic,Gram,Groundnut,Horse-gram,Jowar,Linseed,Maize,Masoor,Moong(Green Gram),Onion,Peas & beans (Pulses),Potato,Ragi,Sesamum,Small millets,Sugarcane,Turmeric,Urad,Wheat"
crop_yield[6] = "Arhar/Tur,Bajra,Barley,Coriander,Cotton(lint),Dry chillies,Dry ginger,Garlic,Gram,Groundnut,Horse-gram,Jowar,Linseed,Maize,Masoor,Moong(Green Gram),Onion,Peas & beans (Pulses),Potato,Rapeseed &Mustard, Rice,Sesamum,Small millets,Sugarcane,Turmeric,Urad,Wheat"
crop_yield[7] = "Arhar/Tur,Barley,Gram,Maize,Masoor,Onion,Potato,Ragi,Rapeseed &Mustard,Rice,Sugarcane,Wheat"
crop_yield[8] = "Arhar/Tur,Banana,Cotton(lint),Dry chillies,Dry ginger,Garlic,Groundnut,Jowar,Maize,Potato,Ragi,Rice,Sesamum,Small millets,Sugarcane,Sweet potato,Turmeric,Wheat"
crop_yield[9] = "Arhar/Tur,Bajra,Castor seed,Coriander,Cotton(lint),Dry chillies,Dry ginger,Garlic,Gram,Groundnut,Horse-gram,Jowar,Linseed,Maize,Masoor,Moong(Green Gram),Onion,Potato,Ragi,Rapeseed &Mustard,Rice,Sesamum,Small millets,Sugarcane,Sunflower,Sweet potato,Turmeric,Urad,Wheat"
crop_yield[10] = "Arhar/Tur,Bajra,Barley,Cotton(lint),Gram,Groundnut,Jowar,Linseed,Maize,Masoor,Moong(Green Gram),Peas & beans (Pulses),Rapeseed &Mustard,Rice,Sesamum,Sugarcane', 'Sunflower', 'Urad', 'Wheat"
crop_yield[11] = "Arhar/Tur,Bajra,Banana,Castor seed,Coriander,Cotton(lint),Dry chillies,Dry ginger,Garlic,Gram,Groundnut,Horse-gram,Jowar,Maize,Moong(Green Gram),Onion,Potato,Ragi,Rapeseed &Mustard,Rice,Sesamum,Small millets,Sugarcane,Sunflower,Sweet potato,Turmeric,Urad,Wheat"
crop_yield[12] = "Arhar/Tur,Bajra,Barley,Dry chillies,Dry ginger,Garlic,Gram,Groundnut,Horse-gram,Jowar,Linseed,Maize,Masoor,Moong(Green Gram), Onion,Peas & beans (Pulses),Potato,Ragi,Rapeseed &Mustard,Rice,Sesamum,Small millets,Sugarcane,Sunflower,Turmeric,Urad,Wheat"

function print_state_y(state_id){
	// given the id of the <select> tag as function argument, it inserts <option> tags
	var option_str = document.getElementById(state_id);
	option_str.length=0;
	option_str.options[0] = new Option('Select State','');
	option_str.selectedIndex = 0;
	for (var i=0; i<state_yield.length; i++) {
		option_str.options[option_str.length] = new Option(state_yield[i],state_yield[i]);
	}
}

function print_crop_y(crop_id, crop_index){
	var option_str = document.getElementById(crop_id);
	option_str.length=0;
	option_str.options[0] = new Option('Select Crop','');
	option_str.selectedIndex = 0;
	var crop_arr = crop_yield[crop_index].split(",");
	for (var i=0; i<crop_yield.length; i++) {
		option_str.options[option_str.length] = new Option(crop_arr[i],crop_arr[i]);
	}
}
var state_price = new Array('Andhra Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Orissa', 'Punjab', 'Rajasthan', 'Tamil Nadu', 'Telangana', 'Uttar Pradesh', 'Uttarakhand','West Bengal');

var crop_price = new Array();

crop_price[0] = "";
crop_price[1] =
  "castor | chickpea | cotton | finger millet | groundnut | maize | pearl millet | pigeonpea | rice | seasmum | sorghum | sugarcane | wheat";
crop_price[2] =
  "castor | chickpea | cotton | linseed | maize | pearl millet | pigeonpea | rape and mustard | rice | seasmum | sugarcane | wheat";
crop_price[3] =
  "barley | castor | chickpea | cotton | finger millet | groundnut | linseed | maize | pearl millet | pigeonpea | rape and mustard | rice | seasmum | sorghum | sugarcane | wheat";
crop_price[4] =
  "barley | castor | chickpea | cotton | finger millet | groundnut | linseed | maize | pearl millet | pigeonpea | rape and mustard | rice | seasmum | sorghum | sugarcane | wheat";
crop_price[5] =
  "barley | castor | chickpea | cotton | finger millet | groundnut | maize |  pearl millet | pigeonpea | rape and mustard | rice | seasmum | sorghum | sugarcane | wheat";
crop_price[6] =
  "barley | castor | chickpea | cotton | groundnut | linseed | maize | pearl millet | pigeonpea | rape and mustard | rice | seasmum | sorghum | sugarcane | wheat";
crop_price[7] =
  "barley | chickpea | cotton | finger millet | groundnut | linseed | maize | pearl millet | pigeonpea | rape and mustard | rice | seasmum | sorghum | sugarcane | wheat";
crop_price[8] =
  "barley | castor | chickpea | finger millet | groundnut | linseed | maize | pearl millet | pigeonpea | rape and mustard | rice | seasmum | sorghum | sugarcane | wheat";
crop_price[9] =
  "castor | chickpea | cotton | finger millet | groundnut | linseed | maize | pearl millet | pigeonpea | rape and mustard | rice | seasmum | sorghum | sugarcane | wheat";
crop_price[10] = "cotton | groundnut | pigeonpea | rice | seasmum | sugarcane";
crop_price[11] =
  "barley | castor | chickpea | cotton | finger millet | groundnut | linseed | maize | pearl millet | pigeonpea | rape and mustard | rice | seasmum | sorghum | sugarcane | wheat";
crop_price[12] =
  "castor | chickpea | cotton | finger millet | groundnut | linseed | maize | pearl millet | pigeonpea | rape and mustard | rice | seasmum | sorghum | sugarcane | wheat";
crop_price[13] =
  "castor | chickpea | cotton | finger millet | groundnut | linseed | maize | pearl millet | pigeonpea | rape and mustard | rice | seasmum | sorghum | sugarcane | wheat";
crop_price[14] =
  "barley | castor | chickpea | cotton | finger millet | groundnut | linseed | maize | pearl millet | pigeonpea | rape and mustard | rice | seasmum | sorghum | sugarcane | wheat";
crop_price[15] =
  "barley | castor | chickpea | cotton | finger millet | groundnut | linseed | maize | pearl millet | pigeonpea | rape and mustard | rice | seasmum | sorghum | sugarcane | wheat";
crop_price[16] =
  "barley | castor | chickpea | cotton | finger millet | groundnut | maize | pearl millet | pigeonpea | rice | seasmum | sorghum | sugarcane | wheat";
crop_price[17] =
  "castor | chickpea | cotton | finger millet | groundnut | linseed | maize | pearl millet | pigeonpea | rice | seasmum | sorghum | sugarcane | wheat";
crop_price[18] =
  "barley | castor | chickpea | cotton | finger millet | groundnut | linseed | maize | pearl millet | pigeonpea | rape and mustard | rice | seasmum | sorghum | sugarcane | wheat";
crop_price[19] =
  "barley | castor | chickpea | cotton | finger millet | groundnut | linseed | maize | pearl millet | pigeonpea | rape and mustard | rice | seasmum | sorghum | sugarcane | wheat";
crop_price[20] =
  "barley | chickpeapigeonpea | rape and mustard | rice | sugarcane | wheat";

function print_state_p(state_id){
	// given the id of the <select> tag as function argument, it inserts <option> tags
	var option_str = document.getElementById(state_id);
	option_str.length=0;
	option_str.options[0] = new Option('Select State','');
	option_str.selectedIndex = 0;
	for (var i=0; i<state_price.length; i++) {
		option_str.options[option_str.length] = new Option(state_price[i],state_price[i]);
	}
}

function print_crop_p(crop_id, crop_index){
	var option_str = document.getElementById(crop_id);
	option_str.length=0;
	option_str.options[0] = new Option('Select Crop','');
	option_str.selectedIndex = 0;
	var crop_arr = crop_price[crop_index].split(" | ");
	for (var i=0; i<crop_price.length; i++) {
		option_str.options[option_str.length] = new Option(crop_arr[i],crop_arr[i]);
	}
}
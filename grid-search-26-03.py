
In[ ]:#######################################################################
26-03-2016
GRID GridSearchCV OUTPUT on raw data
# In[ ]:#######################################################################
train.csv

gsearch1.grid_scores_ [mean: 0.83439, std: 0.00843, params: {'max_depth': 3, 'min_child_weight': 1},
					   mean: 0.83448, std: 0.00904, params: {'max_depth': 3, 'min_child_weight': 3},
					   mean: 0.83473, std: 0.00907, params: {'max_depth': 3, 'min_child_weight': 5},
					   mean: 0.82978, std: 0.00488, params: {'max_depth': 5, 'min_child_weight': 1}, 
					   mean: 0.83059, std: 0.00678, params: {'max_depth': 5, 'min_child_weight': 3}, 
					   mean: 0.83206, std: 0.00730, params: {'max_depth': 5, 'min_child_weight': 5}, 
					   mean: 0.82507, std: 0.00939, params: {'max_depth': 7, 'min_child_weight': 1}, 
					   mean: 0.82625, std: 0.00660, params: {'max_depth': 7, 'min_child_weight': 3}, 
					   mean: 0.82894, std: 0.00689, params: {'max_depth': 7, 'min_child_weight': 5}, 
					   mean: 0.81726, std: 0.00982, params: {'max_depth': 9, 'min_child_weight': 1}, 
					   mean: 0.82142, std: 0.00783, params: {'max_depth': 9, 'min_child_weight': 3}, 
					   mean: 0.82340, std: 0.00571, params: {'max_depth': 9, 'min_child_weight': 5}]
gsearch1.best_params_ {'max_depth': 3, 'min_child_weight': 5}
gsearch1.best_score_ 0.834729588835

gsearch1.grid_scores_ [mean: 0.83396, std: 0.01104, params: {'max_depth': 2, 'min_child_weight': 5}, 
					   mean: 0.83420, std: 0.01094, params: {'max_depth': 2, 'min_child_weight': 6}, 
					   mean: 0.83366, std: 0.01053, params: {'max_depth': 2, 'min_child_weight': 7}, 
					   mean: 0.83477, std: 0.01233, params: {'max_depth': 3, 'min_child_weight': 5}, 
					   mean: 0.83481, std: 0.01202, params: {'max_depth': 3, 'min_child_weight': 6}, 
					   mean: 0.83440, std: 0.01248, params: {'max_depth': 3, 'min_child_weight': 7}, 
					   mean: 0.83398, std: 0.01198, params: {'max_depth': 4, 'min_child_weight': 5}, 
					   mean: 0.83341, std: 0.01152, params: {'max_depth': 4, 'min_child_weight': 6}, 
					   mean: 0.83430, std: 0.01219, params: {'max_depth': 4, 'min_child_weight': 7}]
gsearch1.best_params_ {'max_depth': 3, 'min_child_weight': 6}
gsearch1.best_score_ 0.834807538986
gsearch1.grid_scores_ [mean: 0.83452, std: 0.01244, params: {'gamma': 0.0}, 
						mean: 0.83496, std: 0.01170, params: {'gamma': 0.1}, 
						mean: 0.83435, std: 0.01136, params: {'gamma': 0.2}, 
						mean: 0.83419, std: 0.01001, params: {'gamma': 0.3}, 
						mean: 0.83432, std: 0.01099, params: {'gamma': 0.4}]
gsearch1.best_params_ {'gamma': 0.1}
gsearch1.best_score_ 0.83496069717
gsearch1.grid_scores_ [mean: 0.84048, std: 0.00734, params: {'subsample': 0.6, 'colsample_bytree': 0.6}, 
						mean: 0.83954, std: 0.00663, params: {'subsample': 0.7, 'colsample_bytree': 0.6}, 
						mean: 0.84022, std: 0.00623, params: {'subsample': 0.8, 'colsample_bytree': 0.6}, 
						mean: 0.83996, std: 0.00638, params: {'subsample': 0.9, 'colsample_bytree': 0.6}, 
						 mean: 0.83902, std: 0.00826, params: {'subsample': 0.6, 'colsample_bytree': 0.7}, 
						 mean: 0.83987, std: 0.00704, params: {'subsample': 0.7, 'colsample_bytree': 0.7}, 
						 mean: 0.83940, std: 0.00689, params: {'subsample': 0.8, 'colsample_bytree': 0.7}, 
						 mean: 0.84022, std: 0.00622, params: {'subsample': 0.9, 'colsample_bytree': 0.7}, 
						 mean: 0.84001, std: 0.00781, params: {'subsample': 0.6, 'colsample_bytree': 0.8}, 
						 mean: 0.83860, std: 0.00802, params: {'subsample': 0.7, 'colsample_bytree': 0.8}, 
						 mean: 0.83989, std: 0.00676, params: {'subsample': 0.8, 'colsample_bytree': 0.8}, 
						 mean: 0.84025, std: 0.00684, params: {'subsample': 0.9, 'colsample_bytree': 0.8}, 
						 mean: 0.83919, std: 0.00707, params: {'subsample': 0.6, 'colsample_bytree': 0.9}, 
						 mean: 0.83960, std: 0.00687, params: {'subsample': 0.7, 'colsample_bytree': 0.9}, 
						 mean: 0.84103, std: 0.00704, params: {'subsample': 0.8, 'colsample_bytree': 0.9}, 
						 mean: 0.83945, std: 0.00710, params: {'subsample': 0.9, 'colsample_bytree': 0.9}]
gsearch1.best_params_ {'subsample': 0.8, 'colsample_bytree': 0.9}
gsearch1.best_score_ 0.841034616224
gsearch1.grid_scores_ [mean: 0.82963, std: 0.01417, params: {'subsample': 0.75, 'colsample_bytree': 0.85}, 
						mean: 0.83056, std: 0.01469, params: {'subsample': 0.8, 'colsample_bytree': 0.85}, 
						mean: 0.83073, std: 0.01554, params: {'subsample': 0.85, 'colsample_bytree': 0.85}, 
						mean: 0.83028, std: 0.01449, params: {'subsample': 0.75, 'colsample_bytree': 0.9}, 
						mean: 0.82947, std: 0.01480, params: {'subsample': 0.8, 'colsample_bytree': 0.9}, 
						mean: 0.83136, std: 0.01552, params: {'subsample': 0.85, 'colsample_bytree': 0.9}, 
						mean: 0.83055, std: 0.01415, params: {'subsample': 0.75, 'colsample_bytree': 0.95}, 
						mean: 0.83037, std: 0.01479, params: {'subsample': 0.8, 'colsample_bytree': 0.95}, 
						mean: 0.83087, std: 0.01475, params: {'subsample': 0.85, 'colsample_bytree': 0.95}]
gsearch1.best_params_ {'subsample': 0.85, 'colsample_bytree': 0.9}
gsearch1.best_score_ 0.831355476829
gsearch1.grid_scores_ [mean: 0.84287, std: 0.00923, params: {'reg_alpha': 1e-05}, 
						mean: 0.84358, std: 0.00925, params: {'reg_alpha': 0.01}, 
						mean: 0.84366, std: 0.00866, params: {'reg_alpha': 0.1}, 
						mean: 0.84386, std: 0.00859, params: {'reg_alpha': 1}, 
						mean: 0.82916, std: 0.00817, params: {'reg_alpha': 100}]
gsearch1.best_params_ {'reg_alpha': 1}
gsearch1.best_score_ 0.843860737739
gsearch1.grid_scores_ [mean: 0.83590, std: 0.00988, params: {'reg_alpha': 0.5}, 
						mean: 0.83511, std: 0.00996, params: {'reg_alpha': 1}, 
						mean: 0.83670, std: 0.00924, params: {'reg_alpha': 5}, 
						mean: 0.83600, std: 0.00919, params: {'reg_alpha': 10}]
gsearch1.best_params_ {'reg_alpha': 5}
gsearch1.best_score_ 0.83670305227

clf = xgb.XGBClassifier(missing=np.nan,  
						learning_rate =0.09,
						n_estimators=5000,
						max_depth=3,
						min_child_weight=6,
						gamma=0.1,
						subsample=0.85,
						colsample_bytree=0.9,
						reg_alpha=5,
						objective= 'binary:logistic',
						nthread=4,
						scale_pos_weight=1,
						seed=27)




################################################################################################################
GRID SEARCH ON THETA = 1.9
################################################################################################################
gsearch1.grid_scores_ [
						mean: 0.83996, std: 0.00925, params: {'max_depth': 3, 'min_child_weight': 1}, 
						mean: 0.84095, std: 0.00858, params: {'max_depth': 3, 'min_child_weight': 3}, 
						mean: 0.84090, std: 0.00967, params: {'max_depth': 3, 'min_child_weight': 5}, 
						mean: 0.83731, std: 0.01033, params: {'max_depth': 5, 'min_child_weight': 1}, 
						mean: 0.83838, std: 0.00963, params: {'max_depth': 5, 'min_child_weight': 3}, 
						mean: 0.83859, std: 0.00926, params: {'max_depth': 5, 'min_child_weight': 5}, 
						mean: 0.83337, std: 0.01157, params: {'max_depth': 7, 'min_child_weight': 1}, 
						mean: 0.83237, std: 0.01100, params: {'max_depth': 7, 'min_child_weight': 3}, 
						mean: 0.83471, std: 0.01041, params: {'max_depth': 7, 'min_child_weight': 5}, 
						mean: 0.82670, std: 0.01064, params: {'max_depth': 9, 'min_child_weight': 1}, 
						mean: 0.82988, std: 0.01059, params: {'max_depth': 9, 'min_child_weight': 3}, 
						mean: 0.83057, std: 0.01149, params: {'max_depth': 9, 'min_child_weight': 5}]
gsearch1.best_params_ {'max_depth': 3, 'min_child_weight': 3}
gsearch1.best_score_ 0.840946583326

gsearch2.grid_scores_ [
						mean: 0.82955, std: 0.00715, params: {'max_depth': 1, 'min_child_weight': 2}, 
						mean: 0.82955, std: 0.00715, params: {'max_depth': 1, 'min_child_weight': 3}, 
						mean: 0.82955, std: 0.00715, params: {'max_depth': 1, 'min_child_weight': 4}, 
						mean: 0.83764, std: 0.00734, params: {'max_depth': 2, 'min_child_weight': 2}, 
						mean: 0.83771, std: 0.00744, params: {'max_depth': 2, 'min_child_weight': 3}, 
						mean: 0.83750, std: 0.00788, params: {'max_depth': 2, 'min_child_weight': 4}, 
						mean: 0.83864, std: 0.00784, params: {'max_depth': 3, 'min_child_weight': 2}, 
						mean: 0.83849, std: 0.00758, params: {'max_depth': 3, 'min_child_weight': 3}, 
						mean: 0.83884, std: 0.00774, params: {'max_depth': 3, 'min_child_weight': 4}, 
						mean: 0.83830, std: 0.00728, params: {'max_depth': 4, 'min_child_weight': 2}, 
						mean: 0.83817, std: 0.00737, params: {'max_depth': 4, 'min_child_weight': 3}, 
						mean: 0.83825, std: 0.00697, params: {'max_depth': 4, 'min_child_weight': 4}]
gsearch2.best_params_ {'max_depth': 3, 'min_child_weight': 4}
gsearch2.best_score_ 0.838842538923

gsearch2.grid_scores_ [
						mean: 0.84146, std: 0.00726, params: {'gamma': 0.0}, 
						mean: 0.84143, std: 0.00763, params: {'gamma': 0.1}, 
						mean: 0.84200, std: 0.00709, params: {'gamma': 0.2}, 
						mean: 0.84194, std: 0.00759, params: {'gamma': 0.3}, 
						mean: 0.84173, std: 0.00775, params: {'gamma': 0.4}]
gsearch2.best_params_ {'gamma': 0.2}
gsearch2.best_score_ 0.84200499035

gsearch2.grid_scores_ [
					mean: 0.83736, std: 0.00503, params: {'subsample': 0.6, 'colsample_bytree': 0.6}, 
					mean: 0.83767, std: 0.00386, params: {'subsample': 0.7, 'colsample_bytree': 0.6}, 
					mean: 0.83714, std: 0.00510, params: {'subsample': 0.8, 'colsample_bytree': 0.6}, 
					mean: 0.83772, std: 0.00427, params: {'subsample': 0.9, 'colsample_bytree': 0.6}, 
					mean: 0.83655, std: 0.00380, params: {'subsample': 0.6, 'colsample_bytree': 0.7}, 
					mean: 0.83753, std: 0.00464, params: {'subsample': 0.7, 'colsample_bytree': 0.7}, 
					mean: 0.83692, std: 0.00356, params: {'subsample': 0.8, 'colsample_bytree': 0.7}, 
					mean: 0.83733, std: 0.00466, params: {'subsample': 0.9, 'colsample_bytree': 0.7}, 
					mean: 0.83721, std: 0.00400, params: {'subsample': 0.6, 'colsample_bytree': 0.8}, 
					mean: 0.83713, std: 0.00454, params: {'subsample': 0.7, 'colsample_bytree': 0.8}, 
					mean: 0.83709, std: 0.00460, params: {'subsample': 0.8, 'colsample_bytree': 0.8}, 
					mean: 0.83723, std: 0.00462, params: {'subsample': 0.9, 'colsample_bytree': 0.8}, 
					mean: 0.83712, std: 0.00435, params: {'subsample': 0.6, 'colsample_bytree': 0.9}, 
					mean: 0.83665, std: 0.00405, params: {'subsample': 0.7, 'colsample_bytree': 0.9}, 
					mean: 0.83743, std: 0.00426, params: {'subsample': 0.8, 'colsample_bytree': 0.9}, 
					mean: 0.83701, std: 0.00382, params: {'subsample': 0.9, 'colsample_bytree': 0.9}]
gsearch2.best_params_ {'subsample': 0.9, 'colsample_bytree': 0.6}
gsearch2.best_score_ 0.837716971316

gsearch2.grid_scores_ [
					mean: 0.83990, std: 0.00281, params: {'subsample': 0.55, 'colsample_bytree': 0.85}, 
					mean: 0.83872, std: 0.00249, params: {'subsample': 0.6, 'colsample_bytree': 0.85}, 
					mean: 0.84090, std: 0.00268, params: {'subsample': 0.65, 'colsample_bytree': 0.85}, 
					mean: 0.83924, std: 0.00247, params: {'subsample': 0.55, 'colsample_bytree': 0.9}, 
					mean: 0.83929, std: 0.00235, params: {'subsample': 0.6, 'colsample_bytree': 0.9}, 
					mean: 0.84116, std: 0.00317, params: {'subsample': 0.65, 'colsample_bytree': 0.9}, 
					mean: 0.83959, std: 0.00239, params: {'subsample': 0.55, 'colsample_bytree': 0.95}, 
					mean: 0.83960, std: 0.00298, params: {'subsample': 0.6, 'colsample_bytree': 0.95}, 
					mean: 0.84055, std: 0.00298, params: {'subsample': 0.65, 'colsample_bytree': 0.95}]
gsearch2.best_params_ {'subsample': 0.65, 'colsample_bytree': 0.9}
gsearch2.best_score_ 0.841157083579

gsearch2.grid_scores_ [
					mean: 0.84088, std: 0.01113, params: {'reg_alpha': 1e-05}, 
					mean: 0.84086, std: 0.01127, params: {'reg_alpha': 0.01}, 
					mean: 0.84091, std: 0.01128, params: {'reg_alpha': 0.1}, 
					mean: 0.84111, std: 0.01111, params: {'reg_alpha': 1}, 
					mean: 0.82811, std: 0.01085, params: {'reg_alpha': 100}]
gsearch2.best_params_ {'reg_alpha': 1}
gsearch2.best_score_ 0.841114056991

gsearch2.grid_scores_ [
					mean: 0.84067, std: 0.00850, params: {'reg_alpha': 1}, 
					mean: 0.84078, std: 0.00924, params: {'reg_alpha': 5}, 
					mean: 0.84076, std: 0.00898, params: {'reg_alpha': 7}, 
					mean: 0.84025, std: 0.00908, params: {'reg_alpha': 10}, 
					mean: 0.83885, std: 0.00958, params: {'reg_alpha': 20}]
gsearch2.best_params_ {'reg_alpha': 5}
gsearch2.best_score_ 0.8407766712
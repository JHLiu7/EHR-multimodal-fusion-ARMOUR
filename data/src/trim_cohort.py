



def main():


	# save new split

	# with open('cohort/trim_splits_mextract.p', 'wb') as outf:
	#     pk.dump(new_mex, outf)
	    
	# with open('cohort/trim_splits_drg_ms.p', 'wb') as outf:
	#     pk.dump(new_ms, outf)
	    
	# with open('cohort/trim_splits_drg_apr.p', 'wb') as outf:
	#     pk.dump(new_apr, outf)  
	    


class Checker:
    def __init__(self, split, s_df, t_df):
        
        self.split = split
        
        self.train = split['train']
        self.val = split['val']
        self.test = split['test']
        
        self.all = pd.concat([self.train, self.val, self.test])
        
        self.s_hadms = self.get_s_hadms(s_df)
        self.t_hadms = self.get_t_hadms(t_df)
        
        print('struct hadms:', len(self.s_hadms))
        print('text hadms:', len(self.t_hadms))
        print()
        
    def trim_split(self, main_mode='text'):
        if main_mode == 'text':
            new_split = {}
            missing_hadms = {}
            for k, v in self.split.items():
                mask = v['HADM_ID'].isin(self.s_hadms)
                yes = v[mask]
                no = v[~mask]
                
                new_split[k] = yes
                missing_hadms[k] = no['HADM_ID'].tolist()

            
        elif main_mode == 'struct':
            new_split = {}
            missing_hadms = {}
            for k, v in self.split.items():
                mask = v['HADM_ID'].isin(self.t_hadms)
                yes = v[mask]
                no = v[~mask]
                
                new_split[k] = yes
                missing_hadms[k] = no['HADM_ID'].tolist()

            
        return new_split, missing_hadms
        

    def check_all(self):
        print('Cases, Orig, Struct, Text')
        for name, fold in zip(['All', 'Train', 'Val', 'Test'], [self.all, self.train, self.val, self.test]):
            self._check_fold(fold, name)
        print()
        
    def _check_fold(self, fold, name='Train'):
        orig = len(fold)
        struct = len(fold[fold['HADM_ID'].isin(self.s_hadms)])
        text = len(fold[fold['HADM_ID'].isin(self.t_hadms)])
        
        print(f"{name}: {orig}, {struct}, {text}")
        
    def get_s_hadms(self, s_df):
        hadms = []
        for df in s_df:
            tmp = df.index.get_level_values('hadm_id').unique().tolist()
            
            hadms.extend(tmp)
            
        return hadms
    
    def get_t_hadms(self, t_df):
        return t_df.HADM_ID.unique().tolist()
    
    
    
def _print_split_nums(split):
    train = split['train']
    val = split['val']
    test = split['test']
    
    
    for s in [train, val, test]:
        print(len(s))
        
    print()
    

def check_missing_for_mex(missing_mex):
    
    empty_hadms = []
    
    for name, hadms in missing_mex.items():
        print('Check', name)
        
        has_note, no_note = 0,0
        for hadm in hadms:
            n = notes[notes.HADM_ID==hadm]
            
            if len(n) > 0:
                has_note += 1
            else:
                no_note += 1
                empty_hadms.append(hadm)
        
        print('later note \ no note')
        print(has_note, no_note)
        
    return empty_hadms
import numpy as np
import re
import sys
sys.path.append( "../base" )
from unit import units
from scipy.interpolate import LinearNDInterpolator
from elements import elements
from configparser import ConfigParser

elements_atomic_numbers = elements.elements_atomic_numbers
elements_atomic_weights = elements.elements_atomic_weights
elements_solar_abundance = elements.elements_solar_abundance

def process_ion_file( ion_list, uniqueinds ):
    ion_list_2 = []
    for txt in ion_list:
        x = re.findall("\d-", txt)
        newtxt = txt
        for j in x:
            newtxt = re.sub(j, j[0] + " -" , newtxt)
        ion_list_2.append(newtxt)
    ion_array = np.array(ion_list_2)
    ion_inds = np.arange(len(ion_array))
    mask = np.char.startswith(ion_array, "##############")
    ion_inds_red = ion_inds[mask]
    ion_inds_red = np.concatenate(([0],ion_inds_red))
    ion_ind_groups = []
    for i,j in enumerate(ion_inds_red):
        if i == len( ion_inds_red ) - 1:
            break
        if ion_inds_red[i + 1] - ion_inds_red[i] == 1:
            continue
        ion_inds_gridpoint = np.arange(ion_inds_red[i] + 1, 
                                       ion_inds_red[i + 1]-2,1)
        ion_ind_groups.append(ion_inds_gridpoint)

    ion_ind_groups = np.array(ion_ind_groups,dtype=object)
    ion_ind_groups = ion_ind_groups[uniqueinds]
    return ion_array, ion_ind_groups
            
def get_mu( ovr_data, ion_list, abn_data, param_grid ):
    uniqueinds = np.unique(param_grid,return_index=True,axis=0)[1]
    abn        = abn_data[uniqueinds]
    ovr        = ovr_data[uniqueinds]
    ion_array, ion_ind_groups = process_ion_file( ion_list, uniqueinds )
    mu_list = []
    free_edens_list = []
    H_ion_frac = []
    for grid_ind in range(len(ion_ind_groups)): 
        ion_frac_str = ion_array[ion_ind_groups[grid_ind]]
        #if len(ion_ind_groups[grid_ind]) == 0:
        #    continue
        #print(ion_frac_str)
        mu_num = 0
        mu_denom = 0
        electron_count = 0
        for j, i in enumerate( elements_atomic_numbers ):
            mask = np.char.startswith(ion_frac_str, i)
            #print(i)
            ind = np.where( mask == True )[0][0]

            string = ion_frac_str[ ind ][11:]
            if i == " Hydrogen":
                string = ion_frac_str[ ind ][11:-65]

            ion_frac = list(map(float, string.split()))
            try:
                twoline = np.char.startswith( ion_frac_str[ ind + 1 ], "  " )
            except IndexError:
                twoline = False
            if twoline:
                ion_frac = ion_frac + list(map(float, ion_frac_str[ ind + 1 ][11:].split()))
            ion_frac = np.array(ion_frac)
            if i == " Hydrogen":
                H_ion_frac.append( 10**ion_frac[ 1 ] )
            free_electrons = np.arange(0, len(ion_frac), 1)
            #print(free_electrons)
            electrons = np.sum( free_electrons * 10**ion_frac )
            
            electron_count += 10**abn[grid_ind][j] * electrons
            mu_num   += 10**abn[grid_ind][j] * ( np.sum( 10**ion_frac ) * elements_atomic_weights[i] )
            mu_denom += 10**abn[grid_ind][j] * ( np.sum( 10**ion_frac ) + electrons )
        free_edens_list.append(electron_count)
        mu_list.append( mu_num / mu_denom )
    return np.array( mu_list ), np.array(free_edens_list), np.array(H_ion_frac)

def load_cloudy( dir, fname, gamma=5/3 ):
    """
    Loads CLOUDY output data into python. Assumed dimensions are T, n_H and metallicity.
    """
    dirname = f"{dir}/{fname}"
    ovr = np.loadtxt(f"{dirname}.ovr",usecols=(1,3,4))#[0::2]
    abn = np.loadtxt(f"{dirname}.abn")

    atomic_weights = np.array( list( elements_atomic_weights.values( ) ) ) 
    abn_grid = 10**abn*atomic_weights
    met_grid = np.log10(np.sum(abn_grid[:,2:],axis=1)/np.sum(abn_grid,axis=1))

    param_grid = np.concatenate((ovr[:,(0,1)],np.round(10*met_grid, 5)[:,np.newaxis]),axis=1, dtype=float)
    uniqueinds = np.unique(param_grid,return_index=True,axis=0)[1]
    ion_text_file = open(f"{dirname}.ion", 'r')
    ion_list = ion_text_file.readlines()

    mu_list, edens_list, h_ion_list = get_mu( ovr, ion_list, abn, param_grid )

    datacoolrate = np.loadtxt(f"{dirname}.cool",usecols=(2,3))#[0::2]
    datacoolrate = datacoolrate[uniqueinds]

    Zsol = np.array( list( elements_solar_abundance.values( ) ) )
    lgZsol = np.log10( np.sum(Zsol[2:])/np.sum(Zsol) )
    Z_Zsol = met_grid - lgZsol

    params = param_grid[uniqueinds]

    met_grid = np.unique( np.round( Z_Zsol, decimals=5 ) )
    
    param_met = np.unique(params[:,2],return_index=True, return_counts=True)
    for j,i in enumerate(param_met[0]):
        params[:,2][params[:,2] == i] = met_grid[j]

    mu_cgs = mu_list * units.mp
    lg_mu_cgs   = np.log10( mu_cgs )
    params_coord = params.copy()
    params_coord[:,0] *= 1 / ( gamma - 1 ) * units.kb / mu_cgs
    params_coord[:,1] *= mu_cgs
    crate = (datacoolrate[:,1])/((params_coord[:,1])**2) #Divide by rho * mu

    hrate = (datacoolrate[:,0])/((params_coord[:,1])**2)

    params_coord[:,(0,1)] = np.log10( params_coord[:,(0,1)] )
    
    return { "coolrate" : crate, "heatrate" : hrate,
             "lg_mu_cgs" : lg_mu_cgs, "edens": edens_list,
             "h_ion": h_ion_list, "params_coord": params_coord }

def write_table_cloudy( cloudy_data, interp_range, filename, **kwargs ):
    params_coord = cloudy_data[ "params_coord" ]
    crate        = cloudy_data[ "coolrate"     ]
    hrate        = cloudy_data[ "heatrate"     ]
    lg_mu_cgs    = cloudy_data[ "lg_mu_cgs"    ]
    #edens_list   = cloudy_data[ "edens"        ]
    #h_ion_list   = cloudy_data[ "h_ion"        ]

    cool_int      = LinearNDInterpolator( params_coord, crate      )
    heat_int      = LinearNDInterpolator( params_coord, hrate      )
    lg_mu_cgs_int = LinearNDInterpolator( params_coord, lg_mu_cgs  )
    #edens_int     = LinearNDInterpolator( params_coord, edens_list )
    #hion_int      = LinearNDInterpolator( params_coord, h_ion_list )

    rect_lg_met_const = interp_range[ 0 ]
    rect_lg_rho_const = interp_range[ 1 ]
    rect_lg_eg_const  = interp_range[ 2 ]

    Met,Rho,Eg  = np.meshgrid( rect_lg_met_const, rect_lg_rho_const, rect_lg_eg_const, indexing='ij' )
    rect_coord  = np.array([Eg.flatten(),
                            Rho.flatten(),                        
                            Met.flatten(),
                            ]).transpose()
    if len( rect_coord ) >= 14400:
        print( f"Warning: interpolation table size may exceed constant memory limit. Try to keep the number of floats below 14400" )
    
    reduce_mu_interp = kwargs.get( "reduce_mu_interp", True )
    if reduce_mu_interp:
        rho_rep = kwargs.get( "rho_rep", 1.6e-24 )
        met_rep = kwargs.get( "met_rep", 1 )
        rhoind = np.argmin( np.abs( 10**rect_lg_rho_const - 1 ) - rho_rep )
        metind = np.argmin( np.abs( 10**rect_lg_met_const - 1 ) - met_rep )
        mu_coord_1D = np.array([rect_lg_eg_const,
                        np.ones( rect_lg_eg_const.shape ) * rect_lg_rho_const[ rhoind ],                        
                        np.ones( rect_lg_eg_const.shape ) * rect_lg_met_const[ metind ],
                        ]).transpose()
        lg_mu_cgs_rect = lg_mu_cgs_int ( mu_coord_1D )

    else:
        lg_mu_cgs_rect = lg_mu_cgs_int ( rect_coord )
    lg_cool_rect   = np.log10( cool_int      ( rect_coord ) )
    lg_heat_rect   = np.log10( heat_int      ( rect_coord ) )
    #lg_edens_rect  = edens_int     ( rect_coord )
    #lg_hion_rect   = hion_int      ( rect_coord )
    lg_lam = lg_cool_rect
    lg_gam = lg_heat_rect

    cfg = ConfigParser(  );
    cfg.optionxform = str;

    cfg[ 'cooling' ] = \
        { 'lg_z'   : ' '.join( [ '%0.4f' % s for s in np.unique( rect_coord[ :,2 ] ) ] ),
          'lg_e'   : ' '.join( [ '%0.4f' % s for s in np.unique( rect_coord[ :,0 ] ) ] ),
          'lg_rho' : ' '.join( [ '%0.4f' % s for s in np.unique( rect_coord[ :,1 ] ) ] ),
          'lg_lam' : ' '.join( [ '%0.4f' % s for s in lg_lam ] ),
          'lg_gam' : ' '.join( [ '%0.4f' % s for s in lg_gam ] ),
          'lg_mu_cgs' : ' '.join( [ '%0.4f' % s for s in lg_mu_cgs_rect ] )  
        }
    with open( filename, 'w' ) as f:
        cfg.write( f );
    return

#dat = load_cloudy( "../../cloudyoutputs/production", "cmz_1_14.5_22" )
#params_coord = dat[ "params_coord" ]
#rho_p = params_coord[ :,1 ]
#eg_p  = params_coord[ :,0 ]
#met_p = params_coord[ :,2 ]
#met_inds = [ 4, 6, 8, 11 ]
#interp_range = [ met_p[ met_inds ],
#                 np.linspace( np.min( rho_p ) + 2, 
#                              np.max( rho_p ) - 2, 60 ),
#                 np.linspace( np.min( eg_p ) + 1.1, 
#                              np.max( eg_p ) - 0.75, 60 ) ]
#write_table_cloudy( dat, interp_range, "cooling_test.dat", reduce_mu_interp=True, rho_rep=1.6e-24, met_rep=1 )


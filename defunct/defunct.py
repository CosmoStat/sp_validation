def gamma_T_tc_spatial(                                                         
    cluster_cat,                                                                
    ra_cat,                                                                     
    dec_cat,                                                                    
    e1_cat,                                                                     
    e2_cat,                                                                     
    w_cat=None,                                                                 
):                                                                              
    """Add docstring.                                                           
                                                                                
    ...                                                                         
                                                                                
    """                                                                         
    n_cluster = len(cluster_cat['ra'])                                          
                                                                                
    h = 0.72                                                                    
    cosmo = cosmology.FlatLambdaCDM(H0=h * 100., Om0=0.3)                       
    arcmin_to_rad = np.pi / 180. / 60.                                          
                                                                                
    R_mpc = cluster_cat['R'] / h                                                
                                                                                
    cat_gal = treecorr.Catalog(                                                 
        ra=ra_cat,                                                              
        dec=dec_cat,                                                            
        g1=e1_cat,                                                              
        g2=e2_cat,                                                              
        w=w_cat,                                                                
        ra_units='degrees',                                                     
        dec_units='degrees',                                                    
    )                                                                           
                                                                                
    meanr = []                                                                  
    meanlogr = []                                                               
    xi = []                                                                     
    xi_im = []                                                                  
    varxi = []                                                                  
    for i in tqdm(range(n_cluster), total=n_cluster):                           
                                                                                
        d_ang = cosmo.angular_diameter_distance(cluster_cat['z'][i]).value      
                                                                                
        R_v_ang = R_mpc[i] / d_ang / arcmin_to_rad                              
                                                                                
        cat_cluster = treecorr.Catalog(                                         
            ra=[cluster_cat['ra'][i]],                                          
            dec=[cluster_cat['dec'][i]],                                        
            ra_units='degrees',                                                 
            dec_units='degrees',                                                
        )                    

                TreeCorrConfig = {
            'ra_units': 'degrees',
            'dec_units': 'degrees',
            'max_sep': 3. * R_v_ang,
            'min_sep': R_v_ang / 8.,
            'sep_units': 'arcminutes',
            'nbins': 20,
        }

        ng = treecorr.NGCorrelation(TreeCorrConfig)

        ng.process(cat_cluster, cat_gal)

        meanr.append(ng.meanr * d_ang * arcmin_to_rad / R_mpc[i])
        meanlogr.append(ng.meanlogr)
        xi.append(ng.xi)
        xi_im.append(ng.xi_im)
        varxi.append(ng.varxi)

    meanr = np.array(meanr)
    meanlogr = np.array(meanlogr)
    xi = np.array(xi)
    xi_im = np.array(xi_im)
    varxi = np.array(varxi)

    return meanr, meanlogr, xi, xi_im, varxi


def gamma_T_tc_spatial_parallel(                                                
    cluster_cat,                                                                
    ra_cat,                                                                     
    dec_cat,                                                                    
    e1_cat,                                                                     
    e2_cat,                                                                     
    w_cat=None,                                                                 
):                                                                              
    """Add docstring.                                                           
                                                                                
    ...                                                                         
                                                                                
    """                                                                         
                                                                                
    def job_func(clust_ra, clust_dec, clust_z, clust_R, cat_gal, cosmo):        
        """Add docstring.                                                       
                                                                                
        ...                                                                     
                                                                                
        """                                                                     
        d_ang = cosmo.angular_diameter_distance(clust_z).value                  
                                                                                
        R_v_ang = clust_R / d_ang / arcmin_to_rad                               
                                                                                
        cat_cluster = treecorr.Catalog(                                         
            ra=[clust_ra],                                                      
            dec=[clust_dec],                                                    
            ra_units='degrees',                                                 
            dec_units='degrees',                                                
        )                                                                       
                                                                                
        TreeCorrConfig = {                                                      
            'ra_units': 'degrees',                                              
            'dec_units': 'degrees',                                             
            'max_sep': 3. * R_v_ang,                                            
            'min_sep': R_v_ang / 8.,                                            
            'sep_units': 'arcminutes',                                          
            'nbins': 20,                                                        
        }                                                                       
                                                                                
        ng = treecorr.NGCorrelation(TreeCorrConfig)                             
                                                                                
        ng.process(cat_cluster, cat_gal)

                return np.array([
            ng.meanr * d_ang * arcmin_to_rad / clust_R,
            ng.meanlogr,
            ng.xi,
            ng.xi_im,
            ng.varxi
        ])

    n_cluster = len(cluster_cat['ra'])

    h = 0.72
    cosmo = cosmology.FlatLambdaCDM(H0=h * 100., Om0=0.3)
    arcmin_to_rad = np.pi / 180. / 60.

    R_mpc = cluster_cat['R'] / h

    cat_gal = treecorr.Catalog(
        ra=ra_cat,
        dec=dec_cat,
        g1=e1_cat,
        g2=e2_cat,
        w=w_cat,
        ra_units='degrees',
        dec_units='degrees',
    )

    meanr = []
    meanlogr = []
    xi = []
    xi_im = []
    varxi = []

    res = Parallel(n_jobs=5, backend='loky')(delayed(job_func)(
        cluster_cat['ra'][i],
        cluster_cat['dec'][i],
        cluster_cat['z'][i],
        R_mpc[i],
        cat_gal,
        cosmo
    ) for i in tqdm(range(n_cluster), total=n_cluster))

    return res


def get_theo_gamT(
    cluster_mass,
    cluster_z,
    z_source=0.65,
    H0=72.,
    Omega_m=0.3,
    Omega_b=0.045,
    delta_mdef=500,
    theta=None,
    z_distrib=None,
    meanbin_z=None,
    nz=None,
):
    """Add docstring.

    ...

    """

    def heaviside(a, trunc=0):
        """Add docstring.

        ...

        """
        out = np.zeros_like(a)
        ind = np.where(a > trunc)
        out[ind] = 1
        return out

    def get_z_norm(nz):
        """Add docstring.

        ...

        """
        nz, bins_z = np.histogram(
            nz,
            500,
            density=True,
            range=(0., np.max(nz)),
        )
        meanbin_z = np.mean((bins_z[1:], bins_z[:-1]), 0)
        int_nz = simps(nz, meanbin_z)

        return int_nz

    def get_z_norm2(meanbin_z, nz):
        """Add docstring.

        ...

        """
        int_nz = simps(nz, meanbin_z)

        return int_nz

    def get_norm_gamT(nz, z_min, z_norm=1):
        """Add docstring.

        ...

        """
        nz, bins_z = np.histogram(
            nz,
            500,
            density=True,
            range=(0, np.max(nz)),
        )
        meanbin_z = np.mean((bins_z[1:], bins_z[:-1]), 0)
        trunc_ind = np.where(meanbin_z > z_min)[0]
        int_nz = simps(nz[trunc_ind], meanbin_z[trunc_ind])
        z_mean = simps(
            nz[trunc_ind] * meanbin_z[trunc_ind],
            meanbin_z[trunc_ind]
        ) / int_nz

        return int_nz / z_norm, z_mean

    def get_norm_gamT2(meanbin_z, nz, z_min, z_norm=1):
        """Add docstring.

        ...

        """
        trunc_ind = np.where(meanbin_z > z_min)[0]
        int_nz = simps(nz[trunc_ind], meanbin_z[trunc_ind])
        z_mean = simps(
            nz[trunc_ind] * meanbin_z[trunc_ind],
            meanbin_z[trunc_ind]) / int_nz

        return int_nz / z_norm, z_mean

    def get_gt_model(
        m,
        concentration,
        cluster_z,
        z_distrib,
        cosmo,
        n_bins=500,
    ):
        """Add docstring.

        ...

        """
        nz, bins_z = np.histogram(
            z_distrib,
            n_bins,
            density=True,
            range=(0, np.max(z_distrib)),
        )
        meanbin_z = np.mean((bins_z[1:], bins_z[:-1]), 0)
        gt_models = []
        for i in range(n_bins):
            gt_tmp = clmm.predict_reduced_tangential_shear(
                R * H0 / 100.,
                m,
                concentration,
                cluster_z,
                meanbin_z[i],
                cosmo,
                delta_mdef=delta_mdef,
                halo_profile_model='nfw',
            )
            gt_models.append(gt_tmp)
        gt_models = np.array(gt_models)
        integrand = (
            gt_models * nz.reshape((n_bins, 1))
            * heaviside(meanbin_z, cluster_z).reshape((n_bins, 1))
        )

        final_gt = np.array([
            simps(integrand[:, i], meanbin_z)
            for i in range(integrand.shape[1])
        ])
        return final_gt

    def get_gt_model2(
        m,
        concentration,
        cluster_z,
        meanbin_z,
        nz,
        cosmo,
        n_bins=500,
    ):
        """Add docstring.

        ...

        """
        f = interp1d(meanbin_z, nz)
        zz = np.arange(
            meanbin_z.min(),
            meanbin_z.max(),
            (meanbin_z.max() - meanbin_z.min()) / n_bins,
        )
        gt_models = []
        for i in range(n_bins):
            gt_tmp = clmm.predict_reduced_tangential_shear(
                R * H0 / 100.,
                m,
                concentration,
                cluster_z,
                zz[i],
                cosmo,
                delta_mdef=delta_mdef,
                halo_profile_model='nfw',
            )
            gt_models.append(gt_tmp)
        gt_models = np.array(gt_models)
        integrand = (
            gt_models * f(zz).reshape((n_bins, 1))
            * heaviside(zz, cluster_z).reshape((n_bins, 1))
        )

        final_gt = np.array([
            simps(integrand[:, i], zz) for i in range(integrand.shape[1])
        ])
        return final_gt

    arcmin_to_rad = np.pi / 180. / 60.

    cluster_concentration = c_XRAY(cluster_z, cluster_mass, h=H0 / 100.)

    astropy_cosmo = cosmology.FlatLambdaCDM(H0=H0, Om0=Omega_m, Ob0=Omega_b)

    # cosmo_ccl = cm.cclify_astropy_cosmo(astropy_cosmo)
    cosmo = Cosmology(H0=100 * h, Omega_dm0=Om - Ob, Omega_b0=Ob, Omega_k0=0)

    d_ang = astropy_cosmo.angular_diameter_distance(cluster_z).value            
                                                                                
    R = theta * d_ang * arcmin_to_rad                                           
                                                                                
    if False:                                                                   
        r3d = np.logspace(-2, 2, 100)                                           
    else:                                                                       
        r3d = R                                                                 
                                                                                
    if False:                                                                   
        factor = 1                                                              
    else:                                                                       
        if z_distrib is not None:                                               
            norm_nz = get_z_norm(z_distrib)                                     
            factor, z_mean_src = get_norm_gamT(z_distrib, cluster_z, norm_nz)   
        else:                                                                   
            norm_nz = get_z_norm2(meanbin_z, nz)                                
            factor, z_mean_src = get_norm_gamT2(                                
                meanbin_z,                                                      
                nz,                                                             
                cluster_z,                                                      
                norm_nz,                                                        
            )                                                                   
                                                                                
    if z_distrib is not None:                                                   
        gt = get_gt_model(                                                      
            cluster_mass,                                                       
            cluster_concentration,                                              
            cluster_z,                                                          
            z_distrib,                                                          
            cosmo,                                                              
            n_bins=500,                                                         
        )                                                                       
    else:                                                                       
        gt = get_gt_model2(                                                     
            cluster_mass,                                                       
            cluster_concentration,                                              
            cluster_z,                                                          
            meanbin_z,                                                          
            nz,                                                                 
            cosmo,                                                              
            n_bins=500,                                                         
        )                                                                       
                                                                                
    return gt


def get_theo_mass(
    cluster_z,
    theta,
    gt_profile,
    gt_sig,
    z_source,
    H0=72.,
    Omega_m=0.3,
    Omega_b=0.045,
    z_distrib=None,
    meanbin_z=None,
    nz=None,
    delta_mdef=500,
):
    """Add docstring.

    ...

    """

    def heaviside(a, trunc=0):
        """Add docstring.

        ...

        """
        out = np.zeros_like(a)
        ind = np.where(a > trunc)
        out[ind] = 1
        return out

    def get_z_norm(nz):
        """Add docstring.

        ...

        """
        nz, bins_z = np.histogram(
            nz,
            500,
            density=True,
            range=(0., np.max(nz)),
        )
        meanbin_z = np.mean((bins_z[1:], bins_z[:-1]), 0)
        int_nz = simps(nz, meanbin_z)

        return int_nz

    def get_z_norm2(meanbin_z, nz):
        """Add docstring.

        ...

        """
        int_nz = simps(nz, meanbin_z)

        return int_nz

    def get_norm_gamT(nz, z_min, z_norm=1):
        """Add docstring.

        ...

        """
        nz, bins_z = np.histogram(nz, 500, density=True, range=(0, np.max(nz)))
        meanbin_z = np.mean((bins_z[1:], bins_z[:-1]), 0)
        trunc_ind = np.where(meanbin_z > z_min)[0]
        int_nz = simps(nz[trunc_ind], meanbin_z[trunc_ind])
        z_mean = simps(
            nz[trunc_ind] * meanbin_z[trunc_ind], meanbin_z[trunc_ind]
        ) / int_nz

        return int_nz / z_norm, z_mean

    def get_norm_gamT2(meanbin_z, nz, z_min, z_norm=1):
        """Add docstring.

        ...

        """
        trunc_ind = np.where(meanbin_z > z_min)[0]
        int_nz = simps(nz[trunc_ind], meanbin_z[trunc_ind])
        z_mean = simps(
            nz[trunc_ind] * meanbin_z[trunc_ind], meanbin_z[trunc_ind]
        ) / int_nz

        return int_nz / z_norm, z_mean

    if z_distrib is not None:
        norm_nz = get_z_norm(z_distrib)
        factor, z_mean_src = get_norm_gamT(z_distrib, cluster_z, norm_nz)
    else:
        norm_nz = get_z_norm2(meanbin_z, nz)
        factor, z_mean_src = get_norm_gamT2(meanbin_z, nz, cluster_z, norm_nz)

    arcmin_to_rad = np.pi / 180. / 60.

    astropy_cosmo = cosmology.FlatLambdaCDM(H0=H0, Om0=Omega_m, Ob0=Omega_b)

    # cosmo_ccl = cm.cclify_astropy_cosmo(astropy_cosmo)
    cosmo = Cosmology(H0=100 * h, Omega_dm0=Om - Ob, Omega_b0=Ob, Omega_k0=0)

    def get_gt_model(
        m,
        concentration,
        cluster_z,
        z_distrib,
        cosmo,
        n_bins=500,
    ):
        """Add docstring.

        ...

        """
        nz, bins_z = np.histogram(
            z_distrib,
            n_bins,
            density=True,
            range=(0, np.max(z_distrib)),
        )
        meanbin_z = np.mean((bins_z[1:], bins_z[:-1]), 0)
        gt_models = []
        for i in range(n_bins):
            gt_tmp = clmm.predict_reduced_tangential_shear(
                R * H0 / 100.,
                m,
                concentration,
                cluster_z,
                meanbin_z[i],
                cosmo,
                delta_mdef=delta_mdef,
                halo_profile_model='nfw',
            )
            gt_models.append(gt_tmp)
        gt_models = np.array(gt_models)
        integrand = (
            gt_models * nz.reshape((n_bins, 1))
            * heaviside(meanbin_z, cluster_z).reshape((n_bins, 1))
        )

        final_gt = np.array([
            simps(integrand[:, i], meanbin_z)
            for i in range(integrand.shape[1])
        ])
        return final_gt

     def get_gt_model2(
        m,
        concentration,
        cluster_z,
        meanbin_z,
        nz,
        cosmo,
        n_bins=500,
    ):
        """Add docstring.

        ...

        """
        f = interp1d(meanbin_z, nz)
        zz = np.arange(
            meanbin_z.min(),
            meanbin_z.max(),
            (meanbin_z.max() - meanbin_z.min()) / n_bins,
        )
        gt_models = []
        for i in range(n_bins):
            gt_tmp = clmm.predict_reduced_tangential_shear(
                R * H0 / 100.,
                m,
                concentration,
                cluster_z,
                zz[i],
                cosmo,
                delta_mdef=delta_mdef,
                halo_profile_model='nfw',
            )
            gt_models.append(gt_tmp)
        gt_models = np.array(gt_models)
        integrand = (
            gt_models * f(zz).reshape((n_bins, 1))
            * heaviside(zz, cluster_z).reshape((n_bins, 1))
        )

        final_gt = np.array([
            simps(integrand[:, i], zz) for i in range(integrand.shape[1])
        ])
        return final_gt

    def nfw_to_shear_profile(logm, profile_info):
        [R, gt_profile, gt_sig, z_source] = profile_info
        m = 10. ** logm
        print(m / 1e14)
        concentration = c_XRAY(cluster_z, m, h=H0 / 100.)
        gt_model = get_gt_model(m, concentration, cluster_z, z_source, cosmo)
        return sum((gt_model - gt_profile) ** 2 / gt_sig ** 2)

    def nfw_to_shear_profile2(logm, profile_info):
        [R, gt_profile, gt_sig, meanbin_z, nz] = profile_info
        m = 10. ** logm
        print(m / 1e14)
        concentration = c_XRAY(cluster_z, m, h=H0 / 100.)
        gt_model = get_gt_model2(
            m,
            concentration,
            cluster_z,
            meanbin_z,
            nz,
            cosmo,
        )
        return sum((gt_model - gt_profile) ** 2 / gt_sig ** 2)

    def minimizer(func, x0, args, tolerence=1e-6):
        """Add docstring.

        ...

        """
        minimize_obj = minimize(func, x0, args=args, tol=tolerence)

        return minimize_obj.x[0], minimize_obj.fun, minimize_obj.hess_inv

    def get_best_m(func, args, n_guess=20, tolerence=1e-6, n_jobs=-1):
        """Add docstring.

        ...

        """
        logm_0s = np.random.uniform(13., 15., n_guess)

        res = Parallel(n_jobs=n_jobs, backend='loky')(delayed(minimizer)(
            func,
            x0,
            args,
            tolerence
        ) for x0 in tqdm(logm_0s, total=n_guess))

        res = list((zip(*res)))

        best_values = res[0]
        mini_values = res[1]

                arg_min = np.argmin(mini_values)

        cov_value = (
            np.max((1, np.abs(res[1][arg_min]))) * tolerence * res[2][arg_min]
        )

        return best_values[arg_min], cov_value

    logm_0 = np.random.uniform(13., 15., 1)[0]

    d_ang = astropy_cosmo.angular_diameter_distance(cluster_z).value

    R = theta * d_ang * arcmin_to_rad

    if z_distrib is not None:
        logm_est, cov = get_best_m(
            nfw_to_shear_profile,
            args=[R, gt_profile, gt_sig, z_distrib],
            tolerence=1e-3,
        )
        m_est = 10. ** logm_est
        final_model = get_gt_model(
            m_est,
            c_XRAY(cluster_z, m_est, h=H0 / 100.),
            cluster_z,
            z_distrib,
            cosmo,
            n_bins=100,
        )
    else:
        logm_est, cov = get_best_m(
            nfw_to_shear_profile2,
            args=[R, gt_profile, gt_sig, meanbin_z, nz],
            tolerence=1e-3,
        )
        m_est = 10. ** logm_est
        final_model = get_gt_model2(
            m_est,
            c_XRAY(cluster_z, m_est, h=H0 / 100.),
            cluster_z,
            meanbin_z,
            nz,
            cosmo,
            n_bins=100,
        )

    return m_est, final_model, cov


def stack_mm2(
    ra,
    dec,
    e1,
    e2,
    w,
    cluster_ra,
    cluster_dec,
    cluster_z,
    radius=100,
    n_match=100000,
    tree=None,
):
    """Add docstring.

    ...

    """
    # Project data
    mean_dec = np.mean(dec)
    ra_corr = correct_ra2(ra, dec, mean_dec)
    ra_clust_corr = correct_ra2(cluster_ra, cluster_dec, mean_dec)

    # From Z to comobile
    h = 0.7
    cosmo = cosmology.FlatLambdaCDM(H0=h * 100., Om0=0.3)
    deg_to_rad = np.pi / 180.

    if tree is None:
        tree = cKDTree(np.array([ra_corr, dec]).T)

    k = 0
    for ra_c, dec_c, z_c in tqdm(
        zip(ra_clust_corr, cluster_dec, cluster_z),
        total=len(ra_clust_corr)
    ):

        d_ang = cosmo.angular_diameter_distance(z_c).value   # Rad

        R_max_ang = radius / d_ang / deg_to_rad  # Deg

        res_match = tree.query(np.array([ra_c, dec_c]).T, k=n_match, n_jobs=-1)

        ind_gal = res_match[1][np.where(res_match[0] < R_max_ang)]

        ra_centered = (ra_corr[ind_gal] - ra_c) / R_max_ang
        dec_centered = (dec[ind_gal] - dec_c) / R_max_ang
        if k == 0:
            all_ra = ra_centered
            all_dec = dec_centered
            all_e1 = e1[ind_gal]
            all_e2 = e2[ind_gal]
            all_w = w[ind_gal]
        else:                                                                   
            all_ra = np.concatenate((all_ra, ra_centered))                      
            all_dec = np.concatenate((all_dec, dec_centered))                   
            all_e1 = np.concatenate((all_e1, e1[ind_gal]))                      
            all_e2 = np.concatenate((all_e2, e2[ind_gal]))                      
            all_w = np.concatenate((all_w, w[ind_gal]))                         
                                                                                
        k += 1                                                                  
                                                                                
    return all_ra, all_dec, all_e1, all_e2, all_w


def stack_mm(clust_ra, clust_dec, ra, dec, g1, g2, w, output_size=50):          
    """Add docstring.                                                           
                                                                                
    ...                                                                         
                                                                                
    """                                                                         
    print(len(clust_ra))                                                        
                                                                                
    ke, kb, params = make_mass_map(ra, dec, g1, g2, w)                          
                                                                                
    mean_dec = np.mean(dec)                                                     
    step_ra = params['numbins_ra'] / (params['ra_max'] - params['ra_min'])      
    step_dec = params['numbins_dec'] / (params['dec_max'] - params['dec_min'])  
    ra_tmp = correct_ra2(clust_ra, clust_dec, mean_dec)                         
    ra_pix = params['numbins_ra'] - np.abs(ra_tmp - params['ra_min']) * step_ra 
    dec_pix = (                                                                 
        params['numbins_dec'] - np.abs(clust_dec - params['dec_min'])           
        * step_dec                                                              
    )                                                                           
    plt.figure(figsize=(30, 20))                                                
    plt.imshow(ke)                                                              
    plt.plot(ra_pix, dec_pix, 'k+')                                             
                                                                                
    mean_dec = np.mean(dec)                                                     
    step_ra = params['numbins_ra'] / (params['ra_max'] - params['ra_min'])      
    step_dec = params['numbins_dec'] / (params['dec_max'] - params['dec_min'])  
    stacked_img_e = np.zeros((output_size, output_size))                        
    stacked_img_b = np.zeros((output_size, output_size))                        
    output_size_2 = int(output_size / 2.)                                       
    k = 0                                                                       
    for ra_c, dec_c in zip(clust_ra, clust_dec):                                
        print('{}'.format(k), end='\r')                                         
        ra_tmp = correct_ra2(ra_c, dec_c, mean_dec)                             
        ra_pix = int(                                                           
            params['numbins_ra'] - np.abs(ra_tmp - params['ra_min']) * step_ra  
        )                                                                       
        dec_pix = int(                                                          
            params['numbins_dec'] - np.abs(dec_c - params['dec_min'])           
            * step_dec                                                          
        )                                                                       
        if (                                                                    
            (ra_pix - output_size_2 < 0)                                        
            or (dec_pix - output_size_2 < 0)                                    
            or (ra_pix + output_size_2 > ke.shape[0])                           
            or (dec_pix + output_size_2 > ke.shape[1])                          
        ):                                                                      
            continue                                                            
                                                                                
        stacked_img_e += ke[                                                    
            ra_pix - output_size_2:ra_pix + output_size_2,                      
            dec_pix - output_size_2:dec_pix + output_size_2                     
        ]                                                                       
        stacked_img_b += kb[                                                    
            ra_pix - output_size_2:ra_pix + output_size_2,                      
            dec_pix - output_size_2:dec_pix + output_size_2                     
        ]                                                                       
        k += 1                                                                  
                                                                                
    print(k)                                                                    
                                                                                
    return stacked_img_e, stacked_img_b


def make_maps(map, w, dec, ra, ndec, nra):                                      
    """Make maps.                                                               
                                                                                
    Make tagged maps, weighted by SNR                                           
                                                                                
    """                                                                         
    map_2d, edges = np.histogramdd(                                             
        np.array([dec, ra]).T,                                                  
        bins=(ndec, nra),                                                       
        weights=map * w,                                                        
    )                                                                           
    w_2d, edges = np.histogramdd(                                               
        np.array([dec, ra]).T,                                                  
        bins=(ndec, nra),                                                       
        weights=w,                                                              
    )                                                                           
    w_2d[w_2d == 0] = np.nan                                                    
    return map_2d / w_2d


def make_mass_map(ra, dec, g1, g2, w, proj=True):
    """Add docstring.

    ...

    """
    params = {
        'nside': 2048,  # healpy resolution
        'smooth': 64.,  # smoothing scale to apply (if asked) to the maps
        'jk': 20,  # number of Jack-Knife realizations
        'pix': 0.8,  # angular pixel scale in arcmin
    }

    if proj:
        ra_proj = correct_ra2(ra, dec)  # Projected RA
    else:
        ra_proj = ra
    params['ra_max'] = np.max(ra_proj)
    params['ra_min'] = np.min(ra_proj)
    params['dec_max'] = np.max(dec)
    params['dec_min'] = np.min(dec)
    numbins_ra = max(
        1,
        round((params['ra_max'] - params['ra_min']) * 60. / params['pix'], 0)
    )  # number of bins in ra_l.
    numbins_dec = max(
        1,
        round((params['dec_max'] - params['dec_min']) * 60. / params['pix'], 0)
    )  # number of bins in dec_l.
    params['numbins_ra'] = numbins_ra
    params['numbins_dec'] = numbins_dec

    ellip1 = make_maps(g1, w, dec, ra_proj, numbins_dec, numbins_ra)
    ellip2 = make_maps(g2, w, dec, ra_proj, numbins_dec, numbins_ra)

    ellip1[(ellip1 < np.inf) is False] = 0.
    ellip2[(ellip2 < np.inf) is False] = 0.
    kappa_E, kappa_B = g2k_fft(ellip1, ellip2, 1)

    params['smooth_mode'] = 'tophat'

    smooth_s = get_ang_smoothing(params)
    if params['smooth_mode'] == 'boxcar':
        smooth_kernel = np.ones((smooth_s, smooth_s))
    elif params['smooth_mode'] == 'tophat':
        smooth_kernel = Tophat2DKernel(smooth_s)
    else:
        print("ERROR: Unknown smooth kernel")

    kappa_E_sm = convolve_fft(
        kappa_E,
        smooth_kernel,
        normalize_kernel=np.sum,
        nan_treatment='fill',
    )
    kappa_B_sm = convolve_fft(
        kappa_B,
        smooth_kernel,
        normalize_kernel=np.sum,
        nan_treatment='fill',
    )

    return kappa_E_sm[::-1, ::-1], kappa_B_sm[::-1, ::-1], params


def g2k_fft(g1, g2, dx):                                                        
    """Convert gamma.                                                           
                                                                                
    Convert gamma to kappa in Fourier space                                     
    (Sect. 2.2 from Chang et al. 2016, MNRAS, 459, 3203)                        
                                                                                
    """                                                                         
    g1_3d_ft = fftpack.fft2(g1)                                                 
    g2_3d_ft = fftpack.fft2(g2)                                                 
    FF1 = fftpack.fftfreq(len(g1_3d_ft))                                        
    FF2 = fftpack.fftfreq(len(g1_3d_ft[0]))                                     
                                                                                
    dk = 1.0 / dx * 2 * np.pi                     # max delta_k in 1/arcmin     
    kx = np.dstack(np.meshgrid(FF2, FF1))[:, :, 0] * dk                         
    ky = np.dstack(np.meshgrid(FF2, FF1))[:, :, 1] * dk                         
    kx2 = kx**2                                                                 
    ky2 = ky**2                                                                 
    k2 = kx2 + ky2                                                              
                                                                                
    # kappa to everything                                                       
    k2[k2 == 0] = 1e-15                                                         
    g2kappaE_ft = g1_3d_ft / k2 * (kx2 - ky2) + g2_3d_ft / k2 * 2 * (kx * ky)   
    g2kappaB_ft = (                                                             
        -1 * g1_3d_ft / k2 * 2 * (kx * ky) + g2_3d_ft / k2 * (kx2 - ky2)        
    )                                                                           
                                                                                
    return fftpack.ifft2(g2kappaE_ft).real, fftpack.ifft2(g2kappaB_ft).real


def mad(data, axis=None):                                                       
    """Add docstring.                                                           
                                                                                
    ...                                                                         
                                                                                
    """                                                                         
    return np.median(np.abs(data - np.median(data, axis)), axis) * 1.4826


def weighted_std(x, w):                                                         
    """Add docstring.                                                           
                                                                                
    ...                                                                         
                                                                                
    """                                                                         
    return np.sqrt(                                                             
        len(x) / (len(x) - 1)                                                   
        * np.average((x - np.average(x, weights=w)) ** 2, weights=w)            
    )


def weighted_median(values, weights):                                           
    """Weighted Median.                                                         
                                                                                
    Compute the weighted median of values list.                                 
    The weighted median is computed as follows:                                 
    1- sort both lists (values and weights) based on values.                    
    2- select the 0.5 point from the weights and return the corresponding       
    values as results                                                           
    e.g. values = [1, 3, 0] and weights=[0.1, 0.3, 0.6] assuming weights are    
    probabilities. sorted values = [0, 1, 3] and corresponding sorted           
    weights = [0.6,     0.1, 0.3] the 0.5 point on weight corresponds to the    
    first item which is 0. so the weighted median is 0.                         
                                                                                
    """                                                                         
    # convert the weights into probabilities                                    
    sum_weights = sum(weights)                                                  
    weights = np.array([(w * 1.0) / sum_weights for w in weights])              
    # sort values and weights based on values                                   
    values = np.array(values)                                                   
    sorted_indices = np.argsort(values)                                         
    values_sorted = values[sorted_indices]                                      
    weights_sorted = weights[sorted_indices]                                    
    # select the median point                                                   
    it = np.nditer(weights_sorted, flags=['f_index'])                           
    accumulative_probability = 0                                                
    median_index = -1                                                           
    while not it.finished:                                                      
        accumulative_probability += it[0]                                       
        if accumulative_probability > 0.5:                                      
            median_index = it.index                                             
            return values_sorted[median_index]                                  
        elif accumulative_probability == 0.5:                                   
            median_index = it.index                                             
            it.iternext()                                                       
            next_median_index = it.index                                        
            return np.mean(values_sorted[[median_index, next_median_index]])    
        it.iternext()                                                           
                                                                                
    return values_sorted[median_index]


def get_Isaac_nz(mag, weight, z, alpha=1.79, sigma=1.3, mu=1):                  
    """Add docstring.                                                           
                                                                                
    ...                                                                         
                                                                                
    """                                                                         
                                                                                
    def A(m):                                                                   
        """Add docstring.                                                       
                                                                                
        ...                                                                     
                                                                                
        """                                                                     
        return -0.4154 * m ** 2. + 19.1734 * m - 220.261                        
                                                                                
    def z0(m):                                                                  
        """Add docstring.                                                       
                                                                                
        ...                                                                     
                                                                                
        """                                                                     
        return 0.1081 * m - 1.9417                                              
                                                                                
    m = weighted_median(mag, weight)                                            
                                                                                
    term1 = (                                                                   
        A(m) * (z ** alpha * np.exp(-(z / z0(m)) ** alpha))                     
        / (z0(m) ** (alpha + 1) / alpha * gamma(alpha + 1 / alpha))             
    )                                                                           
    term2 = (1 - A(m)) * np.exp(-(z - mu) ** 2. / (2 * sigma ** 2.)) / 2.5046   
                                                                                
    return term1 + term2


class footprint_mask:                                                           
    """Add docstring.                                                           
                                                                                
    ...                                                                         
                                                                                
    """                                                                         
                                                                                
    def __init__(self):                                                         
        pass                                                                    
                                                                                
    def correct_ra(ra, dec):                                                    
        """Correct RA.                                                          
                                                                                
        It redefines RA according to a sinusoidal projection of the sky,        
        so that all pixels have the same area                                   
        (https://en.wikipedia.org/wiki/Sinusoidal_projection)                   
        """                                                                     
        return (ra - np.mean(ra)) * np.cos(dec / 180. * np.pi)                  
                                                                                
    def binning(self):                                                          
        """Add docstring.                                                       
                                                                                
        ...                                                                     
                                                                                
        """                                                                     
        pass 


def random_cross(                                                               
    n_obj,                                                                      
    ra_cat,                                                                     
    dec_cat,                                                                    
    e1_cat,                                                                     
    e2_cat,                                                                     
    w_cat=None,                                                                 
    n_sample=100,                                                               
):                                                                              
    """Add docstring.                                                           
                                                                                
    ...                                                                         
                                                                                
    """                                                                         
    res = []                                                                    
    for i in tqdm(range(n_sample)):                                             
        cluster_rand = get_random(n_obj)                                        
        res.append(gamma_T_tc(                                                  
            cluster_rand['ra'],                                                 
            cluster_rand['dec'],                                                
            ra_cat,                                                             
            dec_cat,                                                            
            e1_cat,                                                             
            e2_cat,                                                             
            w_cat,                                                              
        ))                                                                      
    res = np.array(res)                                                         
                                                                                
    return res


def get_random(n_obj):                                                          
    """Add docstring.                                                           
                                                                                
    ...                                                                         
                                                                                
    """                                                                         
    final_size = 0                                                              
    while True:                                                                 
        ra = np.random.rand(int(1e6)) * (360. - 0.) + 0.                        
        dec = np.random.rand(int(1e6)) * (90. - 0.) + 0.                        
                                                                                
        m = get_mask_footprint(ra, dec)                                         
                                                                                
        current_size = len(ra[m]) + final_size                                  
                                                                                
        if current_size >= n_obj:                                               
            if final_size == 0:                                                 
                final_ra = ra[m][:n_obj]                                        
                final_dec = dec[m][:n_obj]                                      
                break                                                           
            else:                                                               
                final_ra = np.concatenate(                                      
                    (final_ra, ra[m][:n_obj - final_size])                      
                )                                                               
                final_dec = np.concatenate(                                     
                    (final_dec, dec[m][:n_obj - final_size])                    
                )                                                               
                break                                                           
        else:                                                                   
            if final_size == 0:                                                 
                final_ra = ra[m]                                                
                final_dec = dec[m]                                              
                final_size = len(final_ra)                                      
            else:                                                               
                final_ra = np.concatenate((final_ra, ra[m]))                    
                final_dec = np.concatenate((final_dec, dec[m]))                 
                final_size = len(final_ra)                                      
                                                                                
    return {'ra': final_ra, 'dec': final_dec}


ef match_stars(ra_gal, dec_gal, ra_star, dec_star, thresh=0.0005):             
    """Add docstring.                                                           
                                                                                
    ...                                                                         
                                                                                
    """                                                                         
    # Get right area                                                            
    m_foot_all = get_mask_footprint_W3(ra_gal, dec_gal)                         
    m_foot_star = get_mask_footprint_W3(ra_star, dec_star)                      
                                                                                
    mean_ra = np.mean(ra_gal)                                                   
    mean_dec = np.mean(dec_gal)                                                 
    xx_gal, yy_gal = radec2xy(mean_ra, mean_dec, ra_gal, dec_gal)               
    xx_star, yy_star = radec2xy(mean_ra, mean_dec, ra_star, dec_star)           
                                                                                
    # Make tree with galaxies                                                   
    print("Creating tree...")                                                   
    tree = cKDTree(np.array([xx_gal[m_foot_all], yy_gal[m_foot_all]]).T)        
                                                                                
    # Match stars                                                               
    print("Matching stars...")                                                  
    res_matching = tree.query(                                                  
        np.array([xx_star[m_foot_star], yy_star[m_foot_star]]).T,               
        k=1,                                                                    
        n_jobs=-1,                                                              
    )                                                                           
                                                                                
    # Actual match                                                              
    i_match = np.where(res_matching[0] < thresh * np.pi / 180)                  
                                                                                
    ind_stars = res_matching[1][i_match]                                        
                                                                                
    return ind_stars 


def get_mask_footprint(ra, dec):                                                
    """Add docstring.                                                           
                                                                                
    ...                                                                         
                                                                                
    """                                                                         
    return (                                                                    
        ((ra > 0) & (ra < 39) & (dec > 30.6) & (dec < 36.5))                    
        | ((ra > 341) & (ra < 360) & (dec > 30.6) & (dec < 36.5))               
        | ((ra > 112) & (ra < 200) & (dec > 30.6) & (dec < 40.4))               
        | ((ra > 225.5) & (ra < 270) & (dec > 30.3) & (dec < 40))               
        | ((ra > 197) & (ra < 237.5) & (dec > 54) & (dec < 61.7))               
    )                                                                           
                                                                                
                                                                                
def get_mask_footprint_P1(ra, dec):                                             
    """Add docstring.                                                           
                                                                                
    ...                                                                         
                                                                                
    """                                                                         
    return (ra > 107) & (ra < 171) & (dec > 29) & (dec < 60)                    
                                                                                
                                                                                
def get_mask_footprint_P2(ra, dec):                                             
    """Add docstring.                                                           
                                                                                
    ...                                                                         
                                                                                
    """                                                                         
    return (                                                                    
        (((ra > 0) & (ra < 38)) | ((ra > 340) & (ra < 360)))                    
        & (dec > 29) & (dec < 37)                                               
    ) 

ef get_mask_footprint_P3(ra, dec):                                             
    """Add docstring.                                                           
                                                                                
    ...                                                                         
                                                                                
    """                                                                         
    return (ra > 190) & (ra < 240) & (dec > 47) & (dec < 65)                    
                                                                                
                                                                                
def get_mask_footprint_P4(ra, dec):                                             
    """Add docstring.                                                           
                                                                                
    ...                                                                         
                                                                                
    """                                                                         
    return (ra > 159) & (ra < 201) & (dec > 29) & (dec < 45)                    
                                                                                
                                                                                
def get_mask_footprint_W3(ra, dec):                                             
    """Add docstring.                                                           
                                                                                
    ...                                                                         
                                                                                
    """                                                                         
    return (ra > 208) & (ra < 221) & (dec > 51) & (dec < 58)


def get_mask_footprint_full(ra, dec):                                           
    """Add docstring.                                                           
                                                                                
    ...                                                                         
                                                                                
    """                                                                         
    return (                                                                    
        ((ra > 107) & (ra < 171) & (dec > 29) & (dec < 60))                     
        | (                                                                     
            (((ra > 0) & (ra < 38)) | ((ra > 340) & (ra < 360)))                
            & (dec > 29) & (dec < 37)                                           
        )                                                                       
        | ((ra > 190) & (ra < 240) & (dec > 47) & (dec < 65))                   
        | ((ra > 159) & (ra < 201) & (dec > 29) & (dec < 45))                   
    ) 


def correct_ra(ra, dec, mean_ra=None):                                          
    """Correct RA.                                                              
                                                                                
    It redefines RA according to a sinusoidal projection of the sky,            
    so that all pixels have the same area                                       
    (https://en.wikipedia.org/wiki/Sinusoidal_projection)                       
    """                                                                         
    if mean_ra is None:                                                         
        mean_ra = np.mean(ra)                                                   
                                                                                
    # return (ra - mean_ra)*np.cos(dec/180.*np.pi)                              
    return ra * np.cos(dec / 180. * np.pi)                                      
                                                                                
                                                                                
def correct_ra2(ra, dec, mean_dec=None):                                        
    """Correct RA 2.                                                            
                                                                                
    It redefines RA according to a sinusoidal projection of the sky,            
    so that all pixels have the same area                                       
    (https://en.wikipedia.org/wiki/Sinusoidal_projection)                       
    """                                                                         
    if mean_dec is None:                                                        
        mean_dec = np.mean(dec)                                                 
                                                                                
    # return (ra - mean_ra)*np.cos(dec/180.*np.pi)                              
    return ra * np.cos(mean_dec / 180. * np.pi)


def get_ang_smoothing(params):                                                  
    """Get Angular Smoothing.                                                   
                                                                                
    it translates the smoothing scale to number of pixels.                      
    """                                                                         
    smooth = max(1, int(round(params['smooth'] / params['pix'], 0)))            
    print("smoothing pixel: ", smooth)                                          
    return smooth

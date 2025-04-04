try:
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    Rinstalled = True
except (ImportError, ValueError):
    Rinstalled = False
    print('rpy2 not installed')

def stacked_plot(model_years: list, areas: list):
    SC = 'NewMapping2050Only'
    startDate = '2018-11-15' # 2018 is typically the specified model year in Antares
    endDate   = '2018-12-15'

    if Rinstalled:
        ### 3.1 Load R packages
        importr("base")
        # importr("antaresEditObject")
        importr("antaresViz")

        ### 3.2 Create Production Stack Plots
        # Note that savePlotAsPng requires
        # "PhantomJS" which can be installed
        # by executing: webshot::install_phantomjs()
        
        for i in [0]:
            for year in model_years:
                # Will crash if you have simulations of the same name, only separated in time! (no accounting for time)            

                ## Hourly Profile
                robjects.r('''
                        setSimulationPath(%r, %r)
                        mydata <- readAntares()
                        
                        # Electricity
                        fplot  <- prodStack(mydata, main="%s",
                                    dateRange=c("%s", "%s"),  
                                    areas=c("de00", "dke1", "dkw1"),
                                    interactive=FALSE)
                        savePlotAsPng(fplot, file="Workflow/OverallResults/%s.png", width=800, height=500) # <- note that this requires the PhantomJS
                        
                        # Hydrogen
                        fplot  <- prodStack(mydata, main="%s",
                                    dateRange=c("%s", "%s"),  
                                    areas=c("z_h2_c3_de00", "z_h2_c3_dke1", "z_h2_c3_dkw1"),
                                    interactive=FALSE)
                        savePlotAsPng(fplot, file="Workflow/OverallResults/%s.png", width=800, height=500) # <- note that this requires the PhantomJS
                        '''%tuple(['/Antares'] + 2*[SC + '_Iter%d_Y-%s'%(i, year)] + [startDate, endDate] + [SC + '_Iter%d_Y-%s_elechourly'%(i, year)] +\
                            [SC + '_Iter%d_Y-%s'%(i, year)] + [startDate, endDate] + [SC + '_Iter%d_Y-%s_h2hourly'%(i, year)]))                        
                
                for area in areas:
                    ## Annual Energy
                    robjects.r('''
                                setSimulationPath(%r, %r)
                                mydata <- readAntares(timeStep="annual", areas="%s", links="all")
                                
                                # Electricity
                                fplot  <- prodStack(mydata, main="%s", 
                                            areas="%s",
                                            interactive=FALSE)
                                savePlotAsPng(fplot, file="Workflow/OverallResults/%s.png", width=800, height=1000) # <- note that this requires the PhantomJS
                                
                                # Hydrogen
                                #fplot  <- prodStack(mydata, main="%s", 
                                #            areas="z_h2_c3_%s",
                                #            interactive=FALSE)
                                #savePlotAsPng(fplot, file="Workflow/OverallResults/%s.png", width=800, height=1000) # <- note that this requires the PhantomJS
                                '''%tuple(['/Antares'] + [SC + '_Iter%d_Y-%s'%(i, year)] + [area] +\
                                    [SC + '_Iter%d_Y-%s'%(i, year) + ' - %s'%area] + [area] + [SC + '_Iter%d_Y-%s_%s_elecannual'%(i, year, area)] +\
                                    [SC + '_Iter%d_Y-%s'%(i, year) + ' - %s'%area] + [area] + [SC + '_Iter%d_Y-%s_%s_h2annual'%(i, year, area)]))

                    ## Exchanges (negative is import)
                    robjects.r('''
                                fplot  <- exchangesStack(mydata, 
                                                        area = "%s", 
                                                        main = "Import/Export of %s", 
                                                        unit = "GWh", 
                                                        interactive = FALSE)
                                savePlotAsPng(fplot, file="Workflow/OverallResults/%s.png", width=600, height=1000) # <- note that this requires the PhantomJS
                                '''%(area, area, SC + '_Iter%d_Y-%s_exchanges-%s'%(i, year, area)))
                
    
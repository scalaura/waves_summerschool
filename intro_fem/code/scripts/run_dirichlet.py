from easyFEMpkg.main import run, default_coeffs, default_g

if __name__ == "__main__":
    # coeffs=0, g(x,y)=x+y, geometry='circle'
    run(geometry="circle", max_area=0.02, coeffs=default_coeffs, g=default_g, show_plots=True)

# ----------------------------------------------------------------------------
#
# TITLE - cutout.py
# AUTHOR - James Lane
# PROJECT - tng-dfs
# CONTENTS:
#
# ----------------------------------------------------------------------------

### Docstrings and metadata:
'''Classes and methods for handling TNG cutouts
'''
__author__ = "James Lane"

### Imports
import numpy as np
import warnings
import h5py
import scipy.interpolate
from galpy import orbit
from galpy.util import _rotate_to_arbitrary_vector
_ASTROPY = True
try:
    from astropy import units as apu
except ImportError:
    _ASTROPY = False
from tng_dfs import util

import pdb

_HUBBLE_PARAM = 0.6774 # Planck 2015 value

# ----------------------------------------------------------------------------

# Class to handle methods associated with TNG cutouts
class TNGCutout():
    '''TNGCutout:
    
    Class to handle TNG cutouts
    '''
    
    
    def __init__(self,filename,h=None,z=0.):
        '''__init__:
        
        Initialize a TNGCutout class instance. Hopefully no memory leaks 
        
        Args:
            filename (str) - Filename of file containing simulation data.
            h (float) - Hubble parameter [default 0.7]
            z (float) - redshift of snapshot [default 0.]
        '''
        # If f is a filename then open it. Get header
        self.filename = filename
        with h5py.File(self.filename,'r') as f:
            self.header = dict(f['Header'].attrs.items())
            
        # Set properties
        if h:
            self.h = h
        else:
            _h = self.header['HubbleParam']
            if _h != _HUBBLE_PARAM:
                warnings.warn('Hubble parameter in header is not 0.6774, '+
                              'the Planck 2015 value. Using value in header: '+
                              str(_h))
            self.h = _h
        if z:
            self.z = z
        else:
            self.z = self.header['Redshift']
        self._mass = self.header['MassTable']
        self._npart = self.header['NumPart_ThisFile']
        self._snapnum = self.header['SnapNum']
        self._is_full_snapshot = self._snapnum in _full_snapshot_numbers
        
        self._ptype_fields = {}
        for cr in self.header['CutoutRequest'].split('+'):
            c = cr.split('=')
            self._ptype_fields[c[0]] = c[1].split(',')

        # Centering and rectification flags and info
        self._cen_is_set = False
        self._vcen_is_set = False
        self._rot_is_set = False
        self._cen_ptype = None
        self._cen_mask = None
        self._vcen_ptype = None
        self._vcen_mask = None
        self._rot_ptype = None
        self._rot_mask = None
        self._E_Jcirc_spline = None
            
    def center_and_rectify(self,cen_ptype='PartType4', vcen_ptype='PartType4', 
        rot_ptype='PartType0', cen_scheme='ssc', vcen_scheme='bounded_vcom',
        rot_scheme='bounded_L', **kwargs):
        '''center_and_rectify:
        
        Center the simulation on the primary subhalo and rectify the coordinates 
        such that the disk of the galaxy lies in the XY plane.
        
        Schemes to determine position offset:
            'ssc' - Shrinking-spherical-center algorithm of Power+ 2003
        
        Schemes to determine velocity offset:
            'bounded_com' - Rotate radius-bounded mass-weighted velocity mean
        
        Schemes to determine rotation:
            'bounded_L' - Rotate to radius-bounded angular momentum
        
        Args:
            cen_ptype (str) - Type of particles to use to calculate the 
                position offset of the primary [default PartType4 (star)]
            vcen_ptype (str) - Type of particles to use to calculate the 
                velocity offset of the primary [default PartType4 (gas)]
            rot_ptype (str) - Type of particles to use to calculate the 
                rotation matrix of the primary [default PartType0 (gas)]
            cen_scheme (str) - Scheme to calculate position of primary subhalo
                [default 'ssc']
            vcen_scheme (str) - Scheme to calculate velocity of primary subhalo
                [default 'bounded_vcom']
            rot_scheme (str) - Scheme to calculate rotation matrix of the primary
                [default 'bounded_L']
        
        Returns:
            None, but sets self._cen, self._vcen, and self._rot
        '''    
        # Determine primary subhalo position offset
        cen_coords = self.get_coordinates(cen_ptype, physical=False, 
                                          internal=True)
        cen_masses = self.get_masses(cen_ptype, physical=False, 
                                     internal=True)
        self._cen = self.find_position_center(cen_coords, cen_masses, 
            scheme=cen_scheme, **kwargs)
        self._cen_ptype = cen_ptype
        self._cen_is_set = True
                                          
        # Determine primary subhalo velocity offset
        vcen_coords = self.get_coordinates(vcen_ptype, physical=False, 
                                          internal=True)
        vcen_vels = self.get_velocities(vcen_ptype, physical=False, 
                                        internal=True)
        vcen_masses = self.get_masses(vcen_ptype, physical=False, 
                                      internal=True)
        self._vcen = self.find_velocity_center(vcen_coords, vcen_vels, 
            vcen_masses, scheme=vcen_scheme, **kwargs)
        self._vcen_ptype = vcen_ptype
        self._vcen_is_set = True
        
        # Determine rotation matrix
        rot_coords = self.get_coordinates(rot_ptype, physical=False, 
                                          internal=True)
        rot_vels = self.get_velocities(rot_ptype, physical=False, 
                                        internal=True)
        rot_masses = self.get_masses(rot_ptype, physical=False, 
                                      internal=True)
        self._rot = self.find_rotation_matrix(rot_coords, rot_vels, rot_masses, 
            scheme=rot_scheme, **kwargs)
        self._rot_ptype = rot_ptype
        self._rot_is_set = True

    def get_masses(self,ptype,physical=True,internal=False,key=None,indx=()):
        '''get_masses:
        
        Wrapper for particle mass access, handles conversion to physical units 
        (Msun) and astropy units if necessary
        
        Args:
            ptype (str) - Particle type 
            physical (bool) - Output in physical units (Msun)
            internal (bool) - For internal use in the code, ignore astropy
            indx (list) - List of indices to access, only works for indx of 
                len < 2000. Defaults to all particles (empty list)
            
        Returns:
            masses (np.array) - Masses maybe in physical units (Msun), maybe 
                astropy
        '''
        ptype = util.ptype_to_str(ptype)
        if key is None:
            if ptype == 'PartType1':
                masses = self._mass[1]*np.ones(self._npart[1])
                if len(indx) > 0:
                    masses = masses[indx]
            elif ptype == 'PartType3':
                masses = self._mass[3]*np.ones(self._npart[3])
                if len(indx) > 0:
                    masses = masses[indx]
            else:
                key = 'Masses'
        if key is not None: # Now includes else clause above
            with h5py.File(self.filename,'r') as f:
                if len(indx) < _INDX_MAX_LEN:
                    masses = np.asarray(f[ptype][key][indx])
                else:
                    masses = np.asarray(f[ptype][key])[indx]
        if physical:
            masses = util.mass_code_to_physical(masses,h=self.h,e10=True)
        if _ASTROPY and physical and not internal:
            masses *= apu.M_sun
        return masses
        
    def get_coordinates(self,ptype,physical=True,internal=False,key=None,
        indx=()):
        '''get_coordinates:
        
        Wrapper for particle coordinate access, handles conversion to physical 
        units (kpc) and astropy units if necessary
        
        Args:
            ptype (str) - Particle type 
            physical (bool) - Output in physical units, rather than code units
            internal (bool) - For internal use in the code, ignore astropy
            key (str) - Key to access in particle field. Defaults to 
                'Coordinates' or 'CenterOfMass' to access particle positions, 
                but could be changed to any positional quantity
            indx (list) - List of indices to access, only works for indx of 
                len < 2000. Defaults to all particles (empty list)
        
        Returns:
            coords (np.array) - coords maybe in physical units (kpc), maybe 
                astropy
        '''
        ptype = util.ptype_to_str(ptype)
        if ptype == 'PartType0' and key is None:
            key = 'CenterOfMass'
        elif key is None:
            key = 'Coordinates'
        with h5py.File(self.filename,'r') as f:
            if len(indx) < _INDX_MAX_LEN:
                coords = np.asarray(f[ptype][key][indx])
            else:
                coords = np.asarray(f[ptype][key])[indx]
        if self._cen_is_set:
            coords -= self._cen
        if self._rot_is_set:
            coords = self._apply_rotation_matrix(coords,self._rot)
        if physical:
            coords = util.distance_code_to_physical(coords,h=self.h,z=self.z)
        if _ASTROPY and physical and not internal:
            coords *= apu.kpc
        return coords
    
    def get_velocities(self,ptype,physical=True,internal=False,key=None,
        indx=()):
        '''get_velocities:
        
        Wrapper for particle velocity access, handles conversion to physical 
        units (km/s) and astropy units if necessary
        
        Args:
            ptype (str) - Particle type 
            physical (bool) - Output in physical units, rather than code units
            internal (bool) - For internal use in the code, ignore astropy
            key (str) - Key to access in particle field. Defaults to 
                'Velocities' to access particle velocities, but could be 
                changed to any velocity quantity
            indx (list) - List of indices to access. Defaults to all particles
                (empty list)
        
        Returns:
            vels (np.array) - coords in physical units (km/s), maybe astropy
        '''
        ptype = util.ptype_to_str(ptype)
        if key is None:
            key = 'Velocities'
        with h5py.File(self.filename,'r') as f:
            if len(indx) < _INDX_MAX_LEN:
                vels = np.asarray(f[ptype][key][indx])
            else:
                vels = np.asarray(f[ptype][key])[indx]
        if self._vcen_is_set:
            vels -= self._vcen
        if self._rot_is_set:
            vels = self._apply_rotation_matrix(vels,self._rot)
        if physical:
            vels = util.velocity_code_to_physical(vels,z=self.z)
        if _ASTROPY and physical and not internal:
            vels *= (apu.km/apu.s)
        return vels
    
    def get_potential_energy(self,ptype,physical=True,internal=False,key=None,
        indx=()):
        '''get_potential_energy:
        
        Wrapper for particle potential energy access, handles conversion to 
        physical units (km/s)^2 and astropy units if necessary
        
        Args:
            ptype (str) - Particle type 
            physical (bool) - Output in physical units, rather than code units
            internal (bool) - For internal use in the code, ignore astropy
            key (str) - Key to access in particle field. Defaults to 
                'Potential' to access particle potential energies, but could be 
                changed to any energy quantity such as 'InternalEnergy' for gas
            indx (list) - List of indices to access. Defaults to all particles
                (empty list)
        
        Returns:
            pot (np.array) - potential energy in physical units (km/s)^2, 
                maybe astropy
        '''
        ptype = util.ptype_to_str(ptype)
        if not self._is_full_snapshot:
            raise ValueError('Potential energy is not available for mini snapshots')
        if key is None:
            key = 'Potential'
        with h5py.File(self.filename,'r') as f:
            if len(indx) < _INDX_MAX_LEN:
                pot = np.asarray(f[ptype][key][indx])
            else:
                pot = np.asarray(f[ptype][key])[indx]
        if physical:
            already_physical_key = ['InternalEnergy','InternalEnergyOld','BH_U']
            if key in already_physical_key:
                print('Data for key: '+key+' is already in physical units')
            else:
                pot = util.energy_code_to_physical(pot,z=self.z)
        if _ASTROPY and physical and not internal:
            pot *= (apu.km/apu.s)**2
        return pot
    
    def get_angular_momentum(self,ptype,physical=True,internal=False,indx=()):
        '''get_angular_momentum:

        Wrapper for particle angular momentum access, handles conversion to
        physical units (kpc km/s) and astropy units if necessary

        Args:
            ptype (str) - Particle type
            physical (bool) - Output in physical units, rather than code units
            internal (bool) - For internal use in the code, ignore astropy
            indx (list) - List of indices to access. Defaults to all particles
                (empty list)
        
        Returns:
            L (np.array) - angular momentum in physical units (kpc km/s),
                maybe astropy
        '''
        coords = self.get_coordinates(ptype,physical=physical,internal=internal,
            indx=indx)
        vels = self.get_velocities(ptype,physical=physical,internal=internal,
            indx=indx)
        return np.cross(coords,vels)

    def get_J_Jz_Jp(self,ptype,physical=True,internal=False,indx=()):
        '''get_J_Jz_Jp:
        
        Get the angular momentum magnitude, z-component, and perpendicular
        
        Args:
            ptype (str) - Particle type
            physical (bool) - Output in physical units, rather than code units
            internal (bool) - For internal use in the code, ignore astropy
            indx (list) - List of indices to access. Defaults to all particles
                (empty list)
        
        Returns:
            J,Jz,Jp (np.array) - angular momentum magnitude, z-component, and 
                perpendicular
        '''
        L = self.get_angular_momentum(ptype,physical=physical,internal=internal,
            indx=indx)
        J = np.linalg.norm(L,axis=1)
        Jz = L[:,2]
        Jp = np.sqrt(J**2-Jz**2)
        return J,Jz,Jp

    def get_property(self,ptype,key,numpy_wrap=True,indx=()):
        '''get_property:
        
        Generic wrapper for getting any quantity for a given particle type.
        No unit conversion or astropy output.
        
        Args:
            ptype (str) - Particle type 
            key (str) - Key to access in particle field
            numpy_wrap (bool) - Cast output as a numpy array [default True]
            indx (list) - List of indices to access. Defaults to all particles
                (empty list)
            
        Returns:
            output (unknown) - Output property
        '''
        ptype = util.ptype_to_str(ptype)
        if key not in self._ptype_fields[ptype]:
            raise ValueError('Key: '+key+' not available for particle type: '+ptype)
        with h5py.File(self.filename,'r') as f:
            if len(indx) < _INDX_MAX_LEN:
                output = f[ptype][key][indx]
            else:
                output = f[ptype][key][()][indx]
            if numpy_wrap: # Any reason not to do this?
                output = np.asarray(output)
        return output
    
    def get_orbs(self,ptype,ro=8.,vo=220.,indx=()):
        '''get_orbs:
        
        Turn the particle positions and velocities into a galpy.orbit.Orbit
        object.
        
        Args:
            ptype (str) - Particle type
            
            indx (list) - List of indices to access. Defaults to all particles
                (empty list)
            
        Returns
            orbs (galpy.orbit.Orbit) - Orbit instance
        '''
        # Assume centered and rotated
        if not (self._cen_is_set and self._vcen_is_set and self._rot_is_set):
            raise RuntimeError('Subhalo has not been centered and rotated, '+
                'run center_and_rectify()')
        
        # Make quantities for orbits
        coords = self.get_coordinates(ptype,physical=True,internal=True,
            indx=indx)
        vels = self.get_velocities(ptype,physical=True,internal=True,
            indx=indx)
        
        R = np.sqrt(np.square(coords[:,0]) + np.square(coords[:,1]))
        z = coords[:,2]
        phi = np.arctan2(coords[:,1],coords[:,0])
        vR = (coords[:,0]*vels[:,0] + coords[:,1]*vels[:,1])/R
        vT = (coords[:,0]*vels[:,1] - coords[:,1]*vels[:,0])/R
        vz = vels[:,2]
        
        vxvv = np.array([R/ro,vR/vo,vT/vo,z/ro,vz/vo,phi]).T
        orbs = orbit.Orbit(vxvv,ro=ro,vo=vo)
        return orbs
            
    
    def find_position_center(self,coords,masses,scheme='ssc',**kwargs):
        '''find_position_center:
        
        Wrapper for determining the positional center of the subhalo based on a 
        supplied scheme. Can be 'ssc'
        
        Args:
            coords (np.array) - coordinates in code units
            masses (np.array) - masses in code units
            scheme (str) - Scheme, see above [default 'ssc']
        
        Returns:
            cen (np.array) - length 3 array representing the center of the 
                subhalo in code units.
        '''
        if scheme == 'ssc':
            cen = self._find_position_center_ssc(coords,masses,**kwargs)
        return cen
    
    def _find_position_center_ssc(self,coords,masses,shrink_factor=0.7,
                                  min_particles=100,max_niter=1000,**kwargs):
        '''_find_position_center_ssc:
        
        Use the shrinking-sphere center algorithm of Power+ (2003) to 
        iteratively determine the center of the subhalo. Works by beginning 
        with an initial guess for the center of the subhalo and its size, then 
        determining the COM of the particles within the sphere of radius equal 
        to the starting size. Then iteratively shrink the sphere centered at 
        the calculated COM, and recalculate COM with the particles contained 
        in the sphere. Halt when a minimum number of particles are contained 
        within the sphere.
        
        Args:
            shrink_factor (float) - Factor to shrink the sphere after each 
                iteration of the algorithm [default 0.7]
            min_particles (int) - When this number of particles is contained 
                within the shrinking sphere then terminate the algorithm 
                [default 100]
            max_niter (int) - Maximum number of iterations [default 1000]
        '''
        # Parse kwargs
        if 'shrink_factor' in kwargs:
            shrink_factor = kwargs['shrink_factor']
        if 'min_particles' in kwargs:
            min_particles = kwargs['min_particles']
        if 'max_niter' in kwargs:
            max_niter = kwargs['max_niter']
        
        # Initialize variables
        cen = np.average(coords,axis=0,weights=masses)
        cen_arr = [cen,]
        cen0 = 0
        rs = np.sqrt(np.sum(np.square(coords-cen),axis=1))
        r0 = np.max(rs)
        npart = coords.shape[0]
        
        # Iterate over the algorithm
        niter = 0
        while True:
            # shrink sphere and calculate COM correction
            rs = np.sqrt(np.sum(np.square(coords-cen),axis=1))
            r0 *= shrink_factor
            rs_in_r0 = rs < r0
            cen0 = np.average(coords[rs_in_r0,:]-cen, axis=0, 
                             weights=masses[rs_in_r0])
            if (np.sum(rs_in_r0) < min_particles) or (niter > max_niter):
                break
            else:
                cen += cen0
                cen_arr.append(cen)
                niter += 1
        
        return cen
    
    def find_velocity_center(self,coords,vels,masses,scheme='bounded_vcom',
                             **kwargs):
        '''find_velocity_center:
        
        Wrapper for determining the net velocity of the subhalo based on a 
        supplied scheme. Can be 'bounded_vcom'
        
        Args:
            coords (np.array) - coordinates in code units
            vels (np.array) - velocities in code units
            masses (np.array) - masses in code units
            scheme (str) - Scheme, see above [default 'bounded_vcom']
        
        Returns:
            vcen (np.array) - length 3 array representing the net velocity of 
                the subhalo in code units.
        '''
        if scheme == 'bounded_vcom':
            vcen = self._find_velocity_center_bounded_vcom(coords,vels,masses,
                **kwargs)
        return vcen
    
    def _find_velocity_center_bounded_vcom(self,coords,vels,masses,rmin=0.,
                                          rmax=5.,**kwargs):
        '''_find_velocity_center_bounded_vcom:
        
        Determine the net velocity of the subhalo by calculating the
        mass-weighted velocity of particles in a region bounded by radius.
        
        Args:
            rmin (float) - Minimum radius of particles to consider in kpc
            rmax (float) - Maximum radius of particles to consider in kpc
        '''
        # Parse kwargs, convert to code units
        if 'vcen_rmin' in kwargs:
            rmin = kwargs['vcen_rmin']
        if 'vcen_rmax' in kwargs:
            rmax = kwargs['vcen_rmax']
        rmin = util.distance_physical_to_code(rmin,h=self.h,z=self.z)
        rmax = util.distance_physical_to_code(rmax,h=self.h,z=self.z)
        
        # Find mass-weighted average velocity within radial bounds
        rs = np.sqrt(np.sum(np.square(coords),axis=1))
        vcen_mask = (rs > rmin) & (rs < rmax)
        self._vcen_mask = vcen_mask
        vcen = np.average(vels[vcen_mask,:], axis=0, weights=masses[vcen_mask])
        return vcen
    
    def find_rotation_matrix(self,coords,vels,masses,scheme='bounded_L',
                             **kwargs):
        '''find_rotation_matrix:
        
        Wrapper for determining rotation matrix based on a supplied scheme.
        Can be 'bounded_L'
        
        Args:
            coords (np.array) - coordinates in code units
            vels (np.array) - velocities in code units
            masses (np.array) - masses in code units
            scheme (str) - Scheme, see above [default 'bounded_L']
        
        Returns:
            rot (np.array) - 3x3 array representing the matrix that rotates 
                the 
        '''
        if scheme == 'bounded_L':
            rot = self._find_rotation_matrix_bounded_L(coords,vels,masses,
                **kwargs)
        return rot
    
    def _find_rotation_matrix_bounded_L(self,coords,vels,masses,rmin=5.,
                                        rmax=20.,**kwargs):
        '''_find_rotation_matrix_bounded_L:
        
        Determine the rotation matrix by calculating the total angular momentum 
        of the particles in a region bounded by radius. The rotation matrix 
        rotates the simulation coordinates such that the plane of the galaxy 
        lies in the XY plane.
        
        Args:
            rmin (float) - Minimum radius of particles to consider
            rmax (float) - Maximum radius of particles to consider
        '''
        # Parse kwargs, convert to code units
        if 'rot_rmin' in kwargs:
            rmin = kwargs['rot_rmin']
        if 'rot_rmax' in kwargs:
            rmax = kwargs['rot_rmax']
        rmin = util.distance_physical_to_code(rmin,h=self.h,z=self.z)
        rmax = util.distance_physical_to_code(rmax,h=self.h,z=self.z)
        
        # Calculate angular momentum and rotation
        rs = np.sqrt(np.sum(np.square(coords),axis=1))
        L_mask = (rs > rmin) & (rs < rmax)
        self._rot_mask = L_mask
        L = self._calculate_angular_momentum_vector(coords[L_mask],vels[L_mask],
                                                    masses[L_mask])
        rot = self._calculate_rotation_matrix(L)
        return rot
    
    def _calculate_angular_momentum_vector(self,coords,vels,masses):
        '''_calculate_angular_momentum_vector:
        
        Calculate the total angular momentum of a set of particles
        
        Returns:
            L (np.array) - Length 3 array representing the normalized angular 
                momentum vector.
        '''
        L = masses[:,np.newaxis]*np.cross(coords,vels)
        L = np.sum(L,axis=0)
        L = L/np.sqrt(np.sum(np.square(L)))
        return L
    
    def _calculate_rotation_matrix(self,L,up=np.array([[0.,0.,1.]])):
        '''_calculate_rotation_matrix:
        
        Calculate the matrix that rotates L to up.
        
        Args:
            L (np.array) - length 3 array representing the vector to be rotated
            up (np.array) - Vector to be rotated to (must be shape (3,1))
        
        Returns:
            rot (np.array) - 3x3 array representing the matrix that rotates L 
                to up
        '''
        L = L/np.sqrt(np.sum(np.square(L)))
        rot = _rotate_to_arbitrary_vector(up,L,inv=True)
        # above function requires extra dimension so squeeze
        return np.squeeze(rot)
    
    def _apply_rotation_matrix(self,v,R):
        '''_apply_rotation_matrix:
        
        Apply a rotation matrix
        
        Args:
            v (np.array) - shape (N,3) vector that will be rotated
            R (np.array) - Rotation matrix
        
        Returns:
            vR (np.array) - shape (N,3) rotated vector
        '''
        return np.dot(R,v.T).T
    
    def get_E_Jcirc_spline(self,ptype,angmom='J',scheme='empirical'):
        '''get_E_Jcirc_spline:

        Get a spline that expresses Jcirc as a function of energy based on a 
        supplied scheme. Can be 'empirical'

        Args:
            ptype (str) - Particle type
            angmom (str) - Angular momentum to use, total angular momentum 'J' 
                or angular momentum about rotation axis 'Jz' [default 'J']
            scheme (str) - Scheme to use [default 'empirical']

        Returns:
            None

        Raises:
            RuntimeError - If subhalo has not been centered and rotated
        
        Sets:
            self._E_Jcirc_spline (scipy.interpolate.interp1d) - Spline that
                expresses Jcirc as a function of energy
        '''
        # Assume centered and rotated
        if not (self._cen_is_set and self._vcen_is_set and self._rot_is_set):
            raise RuntimeError('Subhalo has not been centered and rotated')
        
        # Force use of code units so that output is in code units
        vels = self.get_velocities(ptype, physical=False, internal=True)
        L = self.get_angular_momentum(ptype, physical=False, internal=True)
        pot = self.get_potential_energy(ptype, physical=False, internal=True)
        
        # Make energy
        kin = 0.5*np.sum(np.square(vels),axis=1)
        E = kin + pot
        if angmom == 'J':
            J = np.sqrt(np.sum(np.square(L),axis=1))
        elif angmom == 'Jz':
            J = L[:,2]

        # Now make spline based on supplied scheme
        if scheme == 'empirical':
            self._E_Jcirc_spline = self._get_E_Jcirc_spline_empirical(E,J)

        return None
    
    def _get_E_Jcirc_spline_empirical(self,E,J,ngrid=100,spline_fn=None,
            spline_kwargs={'kind':'linear','bounds_error':False,
            'fill_value':'extrapolate'}):
        '''_get_E_Jcirc_spline_empirical:

        Get a spline that expresses Jcirc as a function of energy using 
        an empirical method.

        Args:
            
        Returns:
            E_Jcirc_spline (scipy.interpolate.interp1d) - Spline that
                expresses Jcirc as a function of energy
        '''
        # First bin up E and J to get max J at each E
        E_edges = np.linspace(E.min(),E.max(),num=ngrid+1)
        E_cents = 0.5*(E_edges[1:] + E_edges[:-1])
        J_max = np.zeros(ngrid)
        mask = np.ones(ngrid,dtype=bool)
        for i in range(ngrid):
            E_mask = (E > E_edges[i]) & (E <= E_edges[i+1])
            if np.sum(E_mask) == 0:
                mask[i] = False
                continue
            J_max[i] = np.max(np.abs(J[E_mask])) # Use absolute value incase Jz
        
        # Now make spline
        if spline_fn is None:
            E_Jcirc_spline = scipy.interpolate.interp1d(E_cents[mask],J_max[mask],
                                      **spline_kwargs)
        else:
            E_Jcirc_spline = spline_fn(E_cents[mask],J_max[mask],
                                       **spline_kwargs)
        return E_Jcirc_spline

    def Jcirc(self,E,physical=True,internal=False):
        '''Jcirc:
        
        Calculate the angular momentum of a circular orbit as a function of 
        energy. 
        
        Args:
            E (float) - Energy of orbit (in code units)
            physical (bool) - Output in physical units, rather than code units
            internal (bool) - For internal use in the code, ignore astropy

        Returns:
            Jcirc (float) - Angular momentum of circular orbit
        '''
        if self._E_Jcirc_spline is None:
            raise RuntimeError('E_Jcirc_spline has not been set')
        
        # Check for astropy units
        if isinstance(E,apu.Quantity):
            E = util.energy_physical_to_code(
                E.to(apu.km*apu.km/apu.s/apu.s).value,z=self.z)
        else:
            print('No astropy units detected, assuming code units')

        Jcirc = self._E_Jcirc_spline(E)
        if physical:
            Jcirc = util.angular_momentum_code_to_physical(Jcirc,h=self.h,
                                                           z=self.z)
        if _ASTROPY and physical and not internal:
            Jcirc *= (apu.kpc*apu.km/apu.s)
        return Jcirc
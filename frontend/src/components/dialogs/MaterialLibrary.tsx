import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Tabs,
  Tab,
  Box,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  IconButton,
  Typography,
  Divider,
  Chip,
  Card,
  CardContent,
  Grid
} from '@mui/material';
import {
  Add,
  Edit,
  Delete,
  Search,
  Science,
  Engineering,
  Construction
} from '@mui/icons-material';

interface Material {
  id: string;
  name: string;
  category: 'Steel' | 'Aluminum' | 'Titanium' | 'Composite' | 'Polymer' | 'Ceramic';
  density: number; // kg/m³
  youngsModulus: number; // GPa
  poissonsRatio: number;
  yieldStrength: number; // MPa
  ultimateStrength: number; // MPa
  thermalConductivity: number; // W/m·K
  thermalExpansion: number; // 1/K × 10⁻⁶
  description: string;
  isCustom: boolean;
}

interface MaterialLibraryProps {
  open: boolean;
  onClose: () => void;
  onMaterialSelect?: (material: Material) => void;
}

const predefinedMaterials: Material[] = [
  {
    id: 'steel-structural',
    name: 'Structural Steel (A36)',
    category: 'Steel',
    density: 7850,
    youngsModulus: 200,
    poissonsRatio: 0.26,
    yieldStrength: 250,
    ultimateStrength: 400,
    thermalConductivity: 45,
    thermalExpansion: 12,
    description: 'General purpose structural steel with good weldability',
    isCustom: false
  },
  {
    id: 'steel-stainless',
    name: 'Stainless Steel (304)',
    category: 'Steel',
    density: 8000,
    youngsModulus: 193,
    poissonsRatio: 0.29,
    yieldStrength: 215,
    ultimateStrength: 505,
    thermalConductivity: 16,
    thermalExpansion: 17,
    description: 'Corrosion resistant austenitic stainless steel',
    isCustom: false
  },
  {
    id: 'aluminum-6061',
    name: 'Aluminum Alloy (6061-T6)',
    category: 'Aluminum',
    density: 2700,
    youngsModulus: 68.9,
    poissonsRatio: 0.33,
    yieldStrength: 276,
    ultimateStrength: 310,
    thermalConductivity: 167,
    thermalExpansion: 23,
    description: 'Versatile aluminum alloy with good strength and corrosion resistance',
    isCustom: false
  },
  {
    id: 'titanium-gr2',
    name: 'Titanium (Grade 2)',
    category: 'Titanium',
    density: 4510,
    youngsModulus: 103,
    poissonsRatio: 0.34,
    yieldStrength: 275,
    ultimateStrength: 345,
    thermalConductivity: 17,
    thermalExpansion: 8.6,
    description: 'Commercially pure titanium with excellent corrosion resistance',
    isCustom: false
  }
];

const MaterialLibrary: React.FC<MaterialLibraryProps> = ({ open, onClose, onMaterialSelect }) => {
  const [activeTab, setActiveTab] = useState(0);
  const [materials, setMaterials] = useState<Material[]>(predefinedMaterials);
  const [selectedMaterial, setSelectedMaterial] = useState<Material | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterCategory, setFilterCategory] = useState<string>('All');
  const [newMaterial, setNewMaterial] = useState<Partial<Material>>({
    name: '',
    category: 'Steel',
    density: 0,
    youngsModulus: 0,
    poissonsRatio: 0,
    yieldStrength: 0,
    ultimateStrength: 0,
    thermalConductivity: 0,
    thermalExpansion: 0,
    description: '',
    isCustom: true
  });

  const filteredMaterials = materials.filter(material => {
    const matchesSearch = material.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         material.description.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesCategory = filterCategory === 'All' || material.category === filterCategory;
    return matchesSearch && matchesCategory;
  });

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const handleMaterialSelect = (material: Material) => {
    setSelectedMaterial(material);
  };

  const handleAddMaterial = () => {
    if (newMaterial.name && newMaterial.density && newMaterial.youngsModulus) {
      const material: Material = {
        id: `custom-${Date.now()}`,
        name: newMaterial.name,
        category: newMaterial.category || 'Steel',
        density: newMaterial.density || 0,
        youngsModulus: newMaterial.youngsModulus || 0,
        poissonsRatio: newMaterial.poissonsRatio || 0,
        yieldStrength: newMaterial.yieldStrength || 0,
        ultimateStrength: newMaterial.ultimateStrength || 0,
        thermalConductivity: newMaterial.thermalConductivity || 0,
        thermalExpansion: newMaterial.thermalExpansion || 0,
        description: newMaterial.description || '',
        isCustom: true
      };
      setMaterials([...materials, material]);
      setNewMaterial({
        name: '',
        category: 'Steel',
        density: 0,
        youngsModulus: 0,
        poissonsRatio: 0,
        yieldStrength: 0,
        ultimateStrength: 0,
        thermalConductivity: 0,
        thermalExpansion: 0,
        description: '',
        isCustom: true
      });
    }
  };

  const handleDeleteMaterial = (id: string) => {
    setMaterials(materials.filter(m => m.id !== id));
    if (selectedMaterial?.id === id) {
      setSelectedMaterial(null);
    }
  };

  const handleApplyMaterial = () => {
    if (selectedMaterial && onMaterialSelect) {
      onMaterialSelect(selectedMaterial);
    }
    onClose();
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'Steel': return <Engineering />;
      case 'Aluminum': return <Science />;
      case 'Titanium': return <Construction />;
      default: return <Science />;
    }
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="xl" fullWidth>
      <DialogTitle>
        <Typography variant="h6" component="div" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Science />
          Material Library
        </Typography>
      </DialogTitle>
      <DialogContent>
        <Box sx={{ width: '100%' }}>
          <Tabs value={activeTab} onChange={handleTabChange}>
            <Tab label="Browse Materials" />
            <Tab label="Create Custom Material" />
            <Tab label="Material Properties" />
          </Tabs>

          {activeTab === 0 && (
            <Box sx={{ mt: 2 }}>
              <Grid container spacing={2} sx={{ mb: 3 }}>
                <Grid item xs={8}>
                  <TextField
                    fullWidth
                    placeholder="Search materials..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    InputProps={{
                      startAdornment: <Search sx={{ mr: 1, color: 'text.secondary' }} />
                    }}
                  />
                </Grid>
                <Grid item xs={4}>
                  <FormControl fullWidth>
                    <InputLabel>Category</InputLabel>
                    <Select
                      value={filterCategory}
                      label="Category"
                      onChange={(e) => setFilterCategory(e.target.value)}
                    >
                      <MenuItem value="All">All Categories</MenuItem>
                      <MenuItem value="Steel">Steel</MenuItem>
                      <MenuItem value="Aluminum">Aluminum</MenuItem>
                      <MenuItem value="Titanium">Titanium</MenuItem>
                      <MenuItem value="Composite">Composite</MenuItem>
                      <MenuItem value="Polymer">Polymer</MenuItem>
                      <MenuItem value="Ceramic">Ceramic</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
              </Grid>

              <TableContainer component={Paper} sx={{ maxHeight: 400 }}>
                <Table stickyHeader>
                  <TableHead>
                    <TableRow>
                      <TableCell>Material</TableCell>
                      <TableCell>Category</TableCell>
                      <TableCell align="right">Density (kg/m³)</TableCell>
                      <TableCell align="right">E (GPa)</TableCell>
                      <TableCell align="right">ν</TableCell>
                      <TableCell align="right">Yield (MPa)</TableCell>
                      <TableCell>Actions</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {filteredMaterials.map((material) => (
                      <TableRow
                        key={material.id}
                        hover
                        selected={selectedMaterial?.id === material.id}
                        onClick={() => handleMaterialSelect(material)}
                        sx={{ cursor: 'pointer' }}
                      >
                        <TableCell>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            {getCategoryIcon(material.category)}
                            <Box>
                              <Typography variant="body2" fontWeight="medium">
                                {material.name}
                              </Typography>
                              <Typography variant="caption" color="text.secondary">
                                {material.description}
                              </Typography>
                            </Box>
                          </Box>
                        </TableCell>
                        <TableCell>
                          <Chip
                            label={material.category}
                            size="small"
                            color={material.isCustom ? 'secondary' : 'primary'}
                          />
                        </TableCell>
                        <TableCell align="right">{material.density.toLocaleString()}</TableCell>
                        <TableCell align="right">{material.youngsModulus}</TableCell>
                        <TableCell align="right">{material.poissonsRatio}</TableCell>
                        <TableCell align="right">{material.yieldStrength}</TableCell>
                        <TableCell>
                          <Box sx={{ display: 'flex', gap: 0.5 }}>
                            <IconButton size="small" color="primary">
                              <Edit />
                            </IconButton>
                            {material.isCustom && (
                              <IconButton
                                size="small"
                                color="error"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleDeleteMaterial(material.id);
                                }}
                              >
                                <Delete />
                              </IconButton>
                            )}
                          </Box>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Box>
          )}

          {activeTab === 1 && (
            <Box sx={{ mt: 2 }}>
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    label="Material Name"
                    value={newMaterial.name}
                    onChange={(e) => setNewMaterial({ ...newMaterial, name: e.target.value })}
                    margin="normal"
                  />
                  <FormControl fullWidth margin="normal">
                    <InputLabel>Category</InputLabel>
                    <Select
                      value={newMaterial.category}
                      label="Category"
                      onChange={(e) => setNewMaterial({ ...newMaterial, category: e.target.value as any })}
                    >
                      <MenuItem value="Steel">Steel</MenuItem>
                      <MenuItem value="Aluminum">Aluminum</MenuItem>
                      <MenuItem value="Titanium">Titanium</MenuItem>
                      <MenuItem value="Composite">Composite</MenuItem>
                      <MenuItem value="Polymer">Polymer</MenuItem>
                      <MenuItem value="Ceramic">Ceramic</MenuItem>
                    </Select>
                  </FormControl>
                  <TextField
                    fullWidth
                    label="Description"
                    multiline
                    rows={3}
                    value={newMaterial.description}
                    onChange={(e) => setNewMaterial({ ...newMaterial, description: e.target.value })}
                    margin="normal"
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <Typography variant="h6" gutterBottom>Mechanical Properties</Typography>
                  <TextField
                    fullWidth
                    label="Density (kg/m³)"
                    type="number"
                    value={newMaterial.density}
                    onChange={(e) => setNewMaterial({ ...newMaterial, density: Number(e.target.value) })}
                    margin="normal"
                  />
                  <TextField
                    fullWidth
                    label="Young's Modulus (GPa)"
                    type="number"
                    value={newMaterial.youngsModulus}
                    onChange={(e) => setNewMaterial({ ...newMaterial, youngsModulus: Number(e.target.value) })}
                    margin="normal"
                  />
                  <TextField
                    fullWidth
                    label="Poisson's Ratio"
                    type="number"
                    inputProps={{ step: 0.01, min: 0, max: 0.5 }}
                    value={newMaterial.poissonsRatio}
                    onChange={(e) => setNewMaterial({ ...newMaterial, poissonsRatio: Number(e.target.value) })}
                    margin="normal"
                  />
                  <TextField
                    fullWidth
                    label="Yield Strength (MPa)"
                    type="number"
                    value={newMaterial.yieldStrength}
                    onChange={(e) => setNewMaterial({ ...newMaterial, yieldStrength: Number(e.target.value) })}
                    margin="normal"
                  />
                  <TextField
                    fullWidth
                    label="Ultimate Strength (MPa)"
                    type="number"
                    value={newMaterial.ultimateStrength}
                    onChange={(e) => setNewMaterial({ ...newMaterial, ultimateStrength: Number(e.target.value) })}
                    margin="normal"
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <Typography variant="h6" gutterBottom>Thermal Properties</Typography>
                  <TextField
                    fullWidth
                    label="Thermal Conductivity (W/m·K)"
                    type="number"
                    value={newMaterial.thermalConductivity}
                    onChange={(e) => setNewMaterial({ ...newMaterial, thermalConductivity: Number(e.target.value) })}
                    margin="normal"
                  />
                  <TextField
                    fullWidth
                    label="Thermal Expansion (10⁻⁶/K)"
                    type="number"
                    value={newMaterial.thermalExpansion}
                    onChange={(e) => setNewMaterial({ ...newMaterial, thermalExpansion: Number(e.target.value) })}
                    margin="normal"
                  />
                </Grid>
              </Grid>
              <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end' }}>
                <Button
                  variant="contained"
                  startIcon={<Add />}
                  onClick={handleAddMaterial}
                  disabled={!newMaterial.name || !newMaterial.density || !newMaterial.youngsModulus}
                >
                  Add Material
                </Button>
              </Box>
            </Box>
          )}

          {activeTab === 2 && selectedMaterial && (
            <Box sx={{ mt: 2 }}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    {selectedMaterial.name}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" paragraph>
                    {selectedMaterial.description}
                  </Typography>
                  <Divider sx={{ my: 2 }} />
                  <Grid container spacing={3}>
                    <Grid item xs={12} md={6}>
                      <Typography variant="subtitle1" gutterBottom>Mechanical Properties</Typography>
                      <Box sx={{ '& > div': { display: 'flex', justifyContent: 'space-between', py: 0.5 } }}>
                        <div><span>Density:</span> <strong>{selectedMaterial.density.toLocaleString()} kg/m³</strong></div>
                        <div><span>Young's Modulus:</span> <strong>{selectedMaterial.youngsModulus} GPa</strong></div>
                        <div><span>Poisson's Ratio:</span> <strong>{selectedMaterial.poissonsRatio}</strong></div>
                        <div><span>Yield Strength:</span> <strong>{selectedMaterial.yieldStrength} MPa</strong></div>
                        <div><span>Ultimate Strength:</span> <strong>{selectedMaterial.ultimateStrength} MPa</strong></div>
                      </Box>
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <Typography variant="subtitle1" gutterBottom>Thermal Properties</Typography>
                      <Box sx={{ '& > div': { display: 'flex', justifyContent: 'space-between', py: 0.5 } }}>
                        <div><span>Thermal Conductivity:</span> <strong>{selectedMaterial.thermalConductivity} W/m·K</strong></div>
                        <div><span>Thermal Expansion:</span> <strong>{selectedMaterial.thermalExpansion} × 10⁻⁶/K</strong></div>
                      </Box>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            </Box>
          )}
        </Box>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button
          variant="contained"
          onClick={handleApplyMaterial}
          disabled={!selectedMaterial}
        >
          Apply Material
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default MaterialLibrary;
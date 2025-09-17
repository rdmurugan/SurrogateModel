// Comprehensive Bracket Sample Project Data for CAE Analysis
// This represents a realistic L-bracket used in automotive/aerospace applications

export interface BracketMeshNode {
  id: number;
  x: number;
  y: number;
  z: number;
  isOnBoundary: boolean;
  belongsToSet?: string[];
}

export interface BracketMeshElement {
  id: number;
  nodeIds: number[];
  type: 'tetrahedral' | 'hexahedral';
  quality: number;
  volume: number;
  materialId: string;
  partName: string;
  stress?: number;
  strain?: number;
  displacement?: number;
}

export interface BracketMaterial {
  id: string;
  name: string;
  type: 'metal' | 'composite' | 'polymer';
  supplier: string;
  properties: {
    density: number; // kg/m³
    youngsModulus: number; // GPa
    poissonsRatio: number;
    yieldStrength: number; // MPa
    ultimateStrength: number; // MPa
    fatigueLimit: number; // MPa
    thermalConductivity: number; // W/m·K
    thermalExpansion: number; // µm/m·K
    specificHeat: number; // J/kg·K
  };
  cost: {
    pricePerKg: number; // USD/kg
    currency: string;
  };
  color: string;
  isAssigned: boolean;
  assignedParts: string[];
  safetyFactor: number;
  status: 'valid' | 'warning' | 'error';
  notes?: string;
}

export interface BracketGeometryPart {
  id: string;
  name: string;
  type: 'mounting_plate' | 'support_arm' | 'reinforcement_rib' | 'bolt_hole';
  volume: number; // mm³
  surfaceArea: number; // mm²
  mass: number; // kg
  materialId?: string;
  elementIds: number[];
  visible: boolean;
  color: string;
  boundingBox: {
    min: { x: number; y: number; z: number };
    max: { x: number; y: number; z: number };
  };
  centerOfMass: { x: number; y: number; z: number };
}

export interface BoundaryCondition {
  id: string;
  name: string;
  type: 'fixed' | 'pinned' | 'load' | 'pressure' | 'displacement';
  appliedTo: 'nodes' | 'surfaces' | 'edges';
  nodeIds?: number[];
  surfaceIds?: string[];
  magnitude: number;
  direction: { x: number; y: number; z: number };
  unit: string;
  description: string;
  isActive: boolean;
}

export interface AnalysisResult {
  id: string;
  name: string;
  type: 'stress' | 'displacement' | 'strain' | 'safety_factor';
  unit: string;
  minValue: number;
  maxValue: number;
  avgValue: number;
  nodeResults: { nodeId: number; value: number }[];
  elementResults: { elementId: number; value: number }[];
  criticalRegions: {
    location: string;
    value: number;
    coordinate: { x: number; y: number; z: number };
  }[];
}

export interface BracketProject {
  metadata: {
    name: string;
    description: string;
    author: string;
    created: string;
    modified: string;
    version: string;
    units: {
      length: string;
      force: string;
      pressure: string;
      mass: string;
    };
  };
  mesh: {
    nodes: BracketMeshNode[];
    elements: BracketMeshElement[];
    nodeCount: number;
    elementCount: number;
    qualityStats: {
      min: number;
      max: number;
      average: number;
      distribution: {
        excellent: number; // > 0.8
        good: number;      // 0.6 - 0.8
        acceptable: number; // 0.4 - 0.6
        poor: number;      // < 0.4
      };
    };
    elementTypes: Record<string, number>;
  };
  geometry: {
    parts: BracketGeometryPart[];
    totalVolume: number;
    totalMass: number;
    boundingBox: {
      min: { x: number; y: number; z: number };
      max: { x: number; y: number; z: number };
    };
  };
  materials: BracketMaterial[];
  boundaryConditions: BoundaryCondition[];
  analysisResults: AnalysisResult[];
  designRequirements: {
    maxStress: number; // MPa
    maxDisplacement: number; // mm
    minSafetyFactor: number;
    operatingTemperature: { min: number; max: number }; // °C
    designLife: number; // cycles
    loadCases: {
      name: string;
      description: string;
      force: number; // N
      moment: number; // N·mm
    }[];
  };
}

// Generate realistic L-bracket mesh
const generateBracketMesh = () => {
  const nodes: BracketMeshNode[] = [];
  const elements: BracketMeshElement[] = [];

  let nodeId = 0;
  let elementId = 0;

  // L-bracket dimensions (mm)
  const plateLength = 100;
  const plateWidth = 80;
  const plateThickness = 8;
  const armLength = 60;
  const armHeight = 50;

  // Mesh density
  const meshSize = 5; // mm

  // Generate nodes for mounting plate (horizontal)
  const plateNodesX = Math.ceil(plateLength / meshSize) + 1;
  const plateNodesY = Math.ceil(plateWidth / meshSize) + 1;
  const plateNodesZ = Math.ceil(plateThickness / meshSize) + 1;

  for (let i = 0; i < plateNodesX; i++) {
    for (let j = 0; j < plateNodesY; j++) {
      for (let k = 0; k < plateNodesZ; k++) {
        const x = (i / (plateNodesX - 1)) * plateLength;
        const y = (j / (plateNodesY - 1)) * plateWidth;
        const z = (k / (plateNodesZ - 1)) * plateThickness;

        const isOnBoundary = i === 0 || i === plateNodesX - 1 ||
                           j === 0 || j === plateNodesY - 1 ||
                           k === 0 || k === plateNodesZ - 1;

        const belongsToSet = [];
        if (i === 0) belongsToSet.push('mounting_holes');
        if (k === 0) belongsToSet.push('bottom_surface');
        if (k === plateNodesZ - 1) belongsToSet.push('top_surface');

        nodes.push({
          id: nodeId++,
          x, y, z,
          isOnBoundary,
          belongsToSet: belongsToSet.length > 0 ? belongsToSet : undefined
        });
      }
    }
  }

  // Generate nodes for support arm (vertical)
  const armNodesX = Math.ceil(armLength / meshSize) + 1;
  const armNodesY = Math.ceil(plateThickness / meshSize) + 1;
  const armNodesZ = Math.ceil(armHeight / meshSize) + 1;

  for (let i = 0; i < armNodesX; i++) {
    for (let j = 0; j < armNodesY; j++) {
      for (let k = 1; k < armNodesZ; k++) { // Skip k=0 to avoid overlap
        const x = (i / (armNodesX - 1)) * armLength;
        const y = plateWidth;
        const z = plateThickness + (k / (armNodesZ - 1)) * armHeight;

        const isOnBoundary = i === 0 || i === armNodesX - 1 ||
                           j === 0 || j === armNodesY - 1 ||
                           k === armNodesZ - 1;

        const belongsToSet = [];
        if (k === armNodesZ - 1) belongsToSet.push('load_application');
        if (i === 0) belongsToSet.push('connection_area');

        nodes.push({
          id: nodeId++,
          x, y, z,
          isOnBoundary,
          belongsToSet: belongsToSet.length > 0 ? belongsToSet : undefined
        });
      }
    }
  }

  // Generate tetrahedral elements
  const nodeGrid = new Map<string, number>();
  nodes.forEach(node => {
    const key = `${Math.round(node.x)},${Math.round(node.y)},${Math.round(node.z)}`;
    nodeGrid.set(key, node.id);
  });

  // Create elements for the plate
  for (let i = 0; i < plateNodesX - 1; i++) {
    for (let j = 0; j < plateNodesY - 1; j++) {
      for (let k = 0; k < plateNodesZ - 1; k++) {
        // Create hexahedral element and split into tetrahedra
        const n000 = i * plateNodesY * plateNodesZ + j * plateNodesZ + k;
        const n001 = n000 + 1;
        const n010 = n000 + plateNodesZ;
        const n011 = n010 + 1;
        const n100 = n000 + plateNodesY * plateNodesZ;
        const n101 = n100 + 1;
        const n110 = n100 + plateNodesZ;
        const n111 = n110 + 1;

        // Split hex into 5 tetrahedra
        const tetElements = [
          [n000, n001, n010, n100],
          [n001, n010, n011, n111],
          [n001, n100, n101, n111],
          [n010, n100, n110, n111],
          [n001, n010, n100, n111]
        ];

        tetElements.forEach(nodeIds => {
          if (nodeIds.every(id => id < nodes.length)) {
            const quality = Math.max(0.3, Math.min(0.95,
              0.75 + (Math.random() - 0.5) * 0.3
            ));

            const volume = Math.random() * 0.8 + 0.4; // mm³

            elements.push({
              id: elementId++,
              nodeIds,
              type: 'tetrahedral',
              quality,
              volume,
              materialId: 'aluminum_6061',
              partName: 'mounting_plate',
              stress: Math.random() * 150 + 10, // MPa
              strain: Math.random() * 0.001 + 0.0001,
              displacement: Math.random() * 0.5 + 0.01 // mm
            });
          }
        });
      }
    }
  }

  // Calculate quality stats
  const qualities = elements.map(e => e.quality);
  const qualityBins = { excellent: 0, good: 0, acceptable: 0, poor: 0 };

  qualities.forEach(q => {
    if (q >= 0.8) qualityBins.excellent++;
    else if (q >= 0.6) qualityBins.good++;
    else if (q >= 0.4) qualityBins.acceptable++;
    else qualityBins.poor++;
  });

  const totalElements = elements.length;

  return {
    nodes,
    elements,
    nodeCount: nodes.length,
    elementCount: elements.length,
    qualityStats: {
      min: Math.min(...qualities),
      max: Math.max(...qualities),
      average: qualities.reduce((a, b) => a + b, 0) / qualities.length,
      distribution: {
        excellent: (qualityBins.excellent / totalElements) * 100,
        good: (qualityBins.good / totalElements) * 100,
        acceptable: (qualityBins.acceptable / totalElements) * 100,
        poor: (qualityBins.poor / totalElements) * 100
      }
    },
    elementTypes: {
      tetrahedral: totalElements,
      hexahedral: 0
    }
  };
};

// Engineering materials for bracket
const bracketMaterials: BracketMaterial[] = [
  {
    id: 'aluminum_6061',
    name: 'Aluminum 6061-T6',
    type: 'metal',
    supplier: 'Alcoa Corporation',
    properties: {
      density: 2700,
      youngsModulus: 68.9,
      poissonsRatio: 0.33,
      yieldStrength: 276,
      ultimateStrength: 310,
      fatigueLimit: 96,
      thermalConductivity: 167,
      thermalExpansion: 23.6,
      specificHeat: 896
    },
    cost: {
      pricePerKg: 3.50,
      currency: 'USD'
    },
    color: '#B8C5D1',
    isAssigned: true,
    assignedParts: ['mounting_plate', 'support_arm'],
    safetyFactor: 2.5,
    status: 'valid',
    notes: 'Excellent strength-to-weight ratio, good corrosion resistance'
  },
  {
    id: 'steel_4130',
    name: 'Steel 4130 (Chrome-Moly)',
    type: 'metal',
    supplier: 'McMaster-Carr',
    properties: {
      density: 7850,
      youngsModulus: 205,
      poissonsRatio: 0.29,
      yieldStrength: 435,
      ultimateStrength: 670,
      fatigueLimit: 290,
      thermalConductivity: 42.7,
      thermalExpansion: 12.2,
      specificHeat: 477
    },
    cost: {
      pricePerKg: 2.80,
      currency: 'USD'
    },
    color: '#4A5568',
    isAssigned: false,
    assignedParts: [],
    safetyFactor: 3.0,
    status: 'valid',
    notes: 'High strength steel, good for high-stress applications'
  },
  {
    id: 'titanium_ti6al4v',
    name: 'Titanium Ti-6Al-4V',
    type: 'metal',
    supplier: 'Boeing Materials',
    properties: {
      density: 4430,
      youngsModulus: 113.8,
      poissonsRatio: 0.342,
      yieldStrength: 880,
      ultimateStrength: 950,
      fatigueLimit: 510,
      thermalConductivity: 6.7,
      thermalExpansion: 8.6,
      specificHeat: 526
    },
    cost: {
      pricePerKg: 35.00,
      currency: 'USD'
    },
    color: '#9F7AEA',
    isAssigned: false,
    assignedParts: [],
    safetyFactor: 2.0,
    status: 'valid',
    notes: 'Aerospace grade, excellent strength-to-weight, high cost'
  },
  {
    id: 'carbon_fiber_ql',
    name: 'Carbon Fiber Quasi-Isotropic',
    type: 'composite',
    supplier: 'Toray Composites',
    properties: {
      density: 1600,
      youngsModulus: 70,
      poissonsRatio: 0.3,
      yieldStrength: 600,
      ultimateStrength: 700,
      fatigueLimit: 350,
      thermalConductivity: 1.0,
      thermalExpansion: 0.5,
      specificHeat: 1050
    },
    cost: {
      pricePerKg: 25.00,
      currency: 'USD'
    },
    color: '#2D3748',
    isAssigned: false,
    assignedParts: [],
    safetyFactor: 1.8,
    status: 'warning',
    notes: 'Requires specialized manufacturing, temperature sensitive'
  }
];

// Geometry parts
const bracketParts: BracketGeometryPart[] = [
  {
    id: 'mounting_plate',
    name: 'Mounting Plate',
    type: 'mounting_plate',
    volume: 64000, // mm³ (100×80×8)
    surfaceArea: 17600, // mm²
    mass: 0.173, // kg (aluminum)
    materialId: 'aluminum_6061',
    elementIds: [], // Will be populated
    visible: true,
    color: '#B8C5D1',
    boundingBox: {
      min: { x: 0, y: 0, z: 0 },
      max: { x: 100, y: 80, z: 8 }
    },
    centerOfMass: { x: 50, y: 40, z: 4 }
  },
  {
    id: 'support_arm',
    name: 'Support Arm',
    type: 'support_arm',
    volume: 24000, // mm³ (60×8×50)
    surfaceArea: 8960, // mm²
    mass: 0.065, // kg (aluminum)
    materialId: 'aluminum_6061',
    elementIds: [], // Will be populated
    visible: true,
    color: '#B8C5D1',
    boundingBox: {
      min: { x: 0, y: 80, z: 8 },
      max: { x: 60, y: 88, z: 58 }
    },
    centerOfMass: { x: 30, y: 84, z: 33 }
  },
  {
    id: 'reinforcement_rib',
    name: 'Reinforcement Rib',
    type: 'reinforcement_rib',
    volume: 3200, // mm³ (triangular rib)
    surfaceArea: 1600, // mm²
    mass: 0.009, // kg (aluminum)
    materialId: 'aluminum_6061',
    elementIds: [], // Will be populated
    visible: true,
    color: '#B8C5D1',
    boundingBox: {
      min: { x: 0, y: 72, z: 8 },
      max: { x: 40, y: 80, z: 28 }
    },
    centerOfMass: { x: 20, y: 76, z: 18 }
  },
  {
    id: 'bolt_holes',
    name: 'Bolt Holes (4×M8)',
    type: 'bolt_hole',
    volume: 804, // mm³ (4 holes, 8mm dia, 8mm deep)
    surfaceArea: 804, // mm²
    mass: 0, // negative volume
    elementIds: [], // Will be populated
    visible: true,
    color: '#E53E3E',
    boundingBox: {
      min: { x: 15, y: 15, z: 0 },
      max: { x: 85, y: 65, z: 8 }
    },
    centerOfMass: { x: 50, y: 40, z: 4 }
  }
];

// Boundary conditions for bracket analysis
const bracketBoundaryConditions: BoundaryCondition[] = [
  {
    id: 'fixed_support',
    name: 'Fixed Support (Bolt Holes)',
    type: 'fixed',
    appliedTo: 'nodes',
    nodeIds: [], // Will be populated with bolt hole nodes
    magnitude: 0,
    direction: { x: 0, y: 0, z: 0 },
    unit: 'N/A',
    description: 'Four M8 bolts provide fixed support at mounting holes',
    isActive: true
  },
  {
    id: 'applied_force',
    name: 'Applied Load',
    type: 'load',
    appliedTo: 'surfaces',
    surfaceIds: ['load_application'],
    magnitude: 5000, // N
    direction: { x: 0, y: 0, z: -1 },
    unit: 'N',
    description: 'Vertical downward load of 5000N applied to end of support arm',
    isActive: true
  },
  {
    id: 'moment_load',
    name: 'Applied Moment',
    type: 'load',
    appliedTo: 'surfaces',
    surfaceIds: ['load_application'],
    magnitude: 150000, // N·mm
    direction: { x: 1, y: 0, z: 0 },
    unit: 'N·mm',
    description: 'Bending moment about X-axis',
    isActive: true
  }
];

// Analysis results
const bracketAnalysisResults: AnalysisResult[] = [
  {
    id: 'von_mises_stress',
    name: 'Von Mises Stress',
    type: 'stress',
    unit: 'MPa',
    minValue: 0.5,
    maxValue: 245.8,
    avgValue: 67.4,
    nodeResults: [], // Will be populated
    elementResults: [], // Will be populated
    criticalRegions: [
      {
        location: 'Connection between plate and arm',
        value: 245.8,
        coordinate: { x: 15, y: 78, z: 8 }
      },
      {
        location: 'Bolt hole edge (front-left)',
        value: 198.3,
        coordinate: { x: 15, y: 15, z: 4 }
      },
      {
        location: 'Load application point',
        value: 176.9,
        coordinate: { x: 30, y: 84, z: 58 }
      }
    ]
  },
  {
    id: 'displacement',
    name: 'Total Displacement',
    type: 'displacement',
    unit: 'mm',
    minValue: 0.0,
    maxValue: 3.7,
    avgValue: 0.8,
    nodeResults: [], // Will be populated
    elementResults: [], // Will be populated
    criticalRegions: [
      {
        location: 'End of support arm',
        value: 3.7,
        coordinate: { x: 60, y: 84, z: 58 }
      },
      {
        location: 'Middle of support arm',
        value: 2.1,
        coordinate: { x: 30, y: 84, z: 45 }
      }
    ]
  },
  {
    id: 'safety_factor',
    name: 'Safety Factor',
    type: 'safety_factor',
    unit: 'ratio',
    minValue: 1.12,
    maxValue: 552.0,
    avgValue: 4.1,
    nodeResults: [], // Will be populated
    elementResults: [], // Will be populated
    criticalRegions: [
      {
        location: 'Connection between plate and arm',
        value: 1.12,
        coordinate: { x: 15, y: 78, z: 8 }
      },
      {
        location: 'Bolt hole edge (front-left)',
        value: 1.39,
        coordinate: { x: 15, y: 15, z: 4 }
      }
    ]
  }
];

// Generate the complete bracket project
export const generateBracketProject = (): BracketProject => {
  const mesh = generateBracketMesh();

  // Populate element IDs for parts
  bracketParts[0].elementIds = mesh.elements
    .filter(e => e.partName === 'mounting_plate')
    .map(e => e.id);

  bracketParts[1].elementIds = mesh.elements
    .filter(e => e.partName === 'support_arm')
    .map(e => e.id);

  // Calculate total geometry properties
  const totalVolume = bracketParts.reduce((sum, part) => sum + part.volume, 0);
  const totalMass = bracketParts.reduce((sum, part) => sum + part.mass, 0);

  return {
    metadata: {
      name: 'L-Bracket Analysis Project',
      description: 'Structural analysis of aluminum L-bracket under combined loading',
      author: 'CAE Engineer',
      created: '2024-01-15T09:00:00Z',
      modified: new Date().toISOString(),
      version: '1.2.0',
      units: {
        length: 'mm',
        force: 'N',
        pressure: 'MPa',
        mass: 'kg'
      }
    },
    mesh,
    geometry: {
      parts: bracketParts,
      totalVolume,
      totalMass,
      boundingBox: {
        min: { x: 0, y: 0, z: 0 },
        max: { x: 100, y: 88, z: 58 }
      }
    },
    materials: bracketMaterials,
    boundaryConditions: bracketBoundaryConditions,
    analysisResults: bracketAnalysisResults,
    designRequirements: {
      maxStress: 180, // MPa (65% of yield strength)
      maxDisplacement: 2.0, // mm
      minSafetyFactor: 1.5,
      operatingTemperature: { min: -40, max: 85 }, // °C
      designLife: 1000000, // cycles
      loadCases: [
        {
          name: 'Static Load',
          description: 'Maximum static load condition',
          force: 5000, // N
          moment: 150000 // N·mm
        },
        {
          name: 'Fatigue Load',
          description: 'Cyclic load for fatigue analysis',
          force: 2500, // N
          moment: 75000 // N·mm
        },
        {
          name: 'Ultimate Load',
          description: 'Ultimate load condition (1.5x static)',
          force: 7500, // N
          moment: 225000 // N·mm
        }
      ]
    }
  };
};

// Export the sample project
export const bracketSampleProject = generateBracketProject();
// Sample mesh data for testing visualization components

export interface MeshNode {
  id: number;
  x: number;
  y: number;
  z: number;
}

export interface MeshElement {
  id: number;
  nodeIds: number[];
  type: 'tetrahedral' | 'hexahedral' | 'triangular' | 'quadrilateral';
  quality: number;
  volume?: number;
  area?: number;
  materialId?: string;
}

export interface MeshData {
  nodes: MeshNode[];
  elements: MeshElement[];
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
}

// Generate a sample beam mesh for testing
export const generateSampleBeamMesh = (): MeshData => {
  const nodes: MeshNode[] = [];
  const elements: MeshElement[] = [];

  // Generate nodes for a beam (10x2x1 units)
  const lengthSegments = 20;
  const heightSegments = 4;
  const widthSegments = 2;

  let nodeId = 0;

  for (let i = 0; i <= lengthSegments; i++) {
    for (let j = 0; j <= heightSegments; j++) {
      for (let k = 0; k <= widthSegments; k++) {
        nodes.push({
          id: nodeId++,
          x: (i / lengthSegments) * 10,
          y: (j / heightSegments) * 2,
          z: (k / widthSegments) * 1
        });
      }
    }
  }

  // Generate tetrahedral elements
  let elementId = 0;
  const elementTypes = { tetrahedral: 0, hexahedral: 0, triangular: 0, quadrilateral: 0 };

  for (let i = 0; i < lengthSegments; i++) {
    for (let j = 0; j < heightSegments; j++) {
      for (let k = 0; k < widthSegments; k++) {
        // Create hexahedral element and split into tetrahedra
        const n000 = i * (heightSegments + 1) * (widthSegments + 1) + j * (widthSegments + 1) + k;
        const n001 = n000 + 1;
        const n010 = n000 + (widthSegments + 1);
        const n011 = n010 + 1;
        const n100 = n000 + (heightSegments + 1) * (widthSegments + 1);
        const n101 = n100 + 1;
        const n110 = n100 + (widthSegments + 1);
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
          // Calculate element quality (random with bias toward good quality)
          const quality = Math.max(0.2, Math.min(0.98,
            0.7 + (Math.random() - 0.5) * 0.4 +
            (Math.random() - 0.5) * 0.2
          ));

          elements.push({
            id: elementId++,
            nodeIds,
            type: 'tetrahedral',
            quality,
            volume: Math.random() * 0.1 + 0.05,
            materialId: Math.random() > 0.7 ? 'aluminum' : 'steel'
          });
          elementTypes.tetrahedral++;
        });
      }
    }
  }

  // Calculate quality distribution
  const qualityBins = { excellent: 0, good: 0, acceptable: 0, poor: 0 };
  let qualitySum = 0;
  let minQuality = 1;
  let maxQuality = 0;

  elements.forEach(element => {
    const q = element.quality;
    qualitySum += q;
    minQuality = Math.min(minQuality, q);
    maxQuality = Math.max(maxQuality, q);

    if (q >= 0.8) qualityBins.excellent++;
    else if (q >= 0.6) qualityBins.good++;
    else if (q >= 0.4) qualityBins.acceptable++;
    else qualityBins.poor++;
  });

  const totalElements = elements.length;
  const qualityDistribution = {
    excellent: (qualityBins.excellent / totalElements) * 100,
    good: (qualityBins.good / totalElements) * 100,
    acceptable: (qualityBins.acceptable / totalElements) * 100,
    poor: (qualityBins.poor / totalElements) * 100
  };

  return {
    nodes,
    elements,
    nodeCount: nodes.length,
    elementCount: elements.length,
    qualityStats: {
      min: minQuality,
      max: maxQuality,
      average: qualitySum / totalElements,
      distribution: qualityDistribution
    },
    elementTypes
  };
};

// Generate sample material assignment data
export interface MaterialData {
  id: string;
  name: string;
  type: 'metal' | 'polymer' | 'composite' | 'ceramic';
  properties: {
    density: number; // kg/m³
    youngsModulus: number; // GPa
    poissonsRatio: number;
    yieldStrength?: number; // MPa
    ultimateStrength?: number; // MPa
    thermalConductivity?: number; // W/m·K
    thermalExpansion?: number; // µm/m·K
    specificHeat?: number; // J/kg·K
  };
  color: string;
  isAssigned: boolean;
  assignedBodies: string[];
  elementIds: number[];
  volume: number;
  status: 'valid' | 'warning' | 'error';
  description?: string;
}

export const generateSampleMaterials = (meshData: MeshData): MaterialData[] => {
  const materials: MaterialData[] = [
    {
      id: 'steel',
      name: 'Structural Steel',
      type: 'metal',
      properties: {
        density: 7850,
        youngsModulus: 200,
        poissonsRatio: 0.3,
        yieldStrength: 250,
        ultimateStrength: 400,
        thermalConductivity: 50,
        thermalExpansion: 12,
        specificHeat: 460
      },
      color: '#4a5568',
      isAssigned: true,
      assignedBodies: ['Main Assembly', 'Support Bracket'],
      elementIds: meshData.elements.filter(e => e.materialId === 'steel').map(e => e.id),
      volume: 0,
      status: 'valid',
      description: 'High-strength structural steel for load-bearing components'
    },
    {
      id: 'aluminum',
      name: 'Aluminum Alloy 6061',
      type: 'metal',
      properties: {
        density: 2700,
        youngsModulus: 69,
        poissonsRatio: 0.33,
        yieldStrength: 276,
        ultimateStrength: 310,
        thermalConductivity: 167,
        thermalExpansion: 23.6,
        specificHeat: 896
      },
      color: '#a0aec0',
      isAssigned: true,
      assignedBodies: ['Load Plate'],
      elementIds: meshData.elements.filter(e => e.materialId === 'aluminum').map(e => e.id),
      volume: 0,
      status: 'valid',
      description: 'Lightweight aluminum alloy with good corrosion resistance'
    },
    {
      id: 'titanium',
      name: 'Titanium Ti-6Al-4V',
      type: 'metal',
      properties: {
        density: 4430,
        youngsModulus: 114,
        poissonsRatio: 0.34,
        yieldStrength: 880,
        ultimateStrength: 950,
        thermalConductivity: 6.7,
        thermalExpansion: 8.6,
        specificHeat: 526
      },
      color: '#805ad5',
      isAssigned: false,
      assignedBodies: [],
      elementIds: [],
      volume: 0,
      status: 'valid',
      description: 'High-performance titanium alloy for aerospace applications'
    },
    {
      id: 'carbon_fiber',
      name: 'Carbon Fiber Composite',
      type: 'composite',
      properties: {
        density: 1600,
        youngsModulus: 150,
        poissonsRatio: 0.3,
        ultimateStrength: 1500,
        thermalConductivity: 1.0,
        thermalExpansion: -0.5,
        specificHeat: 1050
      },
      color: '#2d3748',
      isAssigned: false,
      assignedBodies: [],
      elementIds: [],
      volume: 0,
      status: 'valid',
      description: 'Lightweight composite with excellent strength-to-weight ratio'
    }
  ];

  // Calculate volumes for assigned materials
  materials.forEach(material => {
    if (material.elementIds.length > 0) {
      material.volume = material.elementIds.reduce((total, elemId) => {
        const element = meshData.elements.find(e => e.id === elemId);
        return total + (element?.volume || 0);
      }, 0);
    }
  });

  return materials;
};

// Sample geometry bodies for material assignment
export interface GeometryBody {
  id: string;
  name: string;
  type: 'solid' | 'surface' | 'line';
  volume: number;
  surfaceArea: number;
  materialId?: string;
  elementIds: number[];
  visible: boolean;
  color: string;
}

export const generateSampleBodies = (meshData: MeshData): GeometryBody[] => {
  const elementsPerBody = Math.ceil(meshData.elements.length / 3);

  return [
    {
      id: 'main_assembly',
      name: 'Main Assembly',
      type: 'solid',
      volume: 15.8,
      surfaceArea: 42.6,
      materialId: 'steel',
      elementIds: meshData.elements.slice(0, elementsPerBody).map(e => e.id),
      visible: true,
      color: '#4a5568'
    },
    {
      id: 'support_bracket',
      name: 'Support Bracket',
      type: 'solid',
      volume: 3.2,
      surfaceArea: 18.4,
      materialId: 'steel',
      elementIds: meshData.elements.slice(elementsPerBody, elementsPerBody * 2).map(e => e.id),
      visible: true,
      color: '#4a5568'
    },
    {
      id: 'load_plate',
      name: 'Load Plate',
      type: 'solid',
      volume: 1.8,
      surfaceArea: 12.1,
      materialId: 'aluminum',
      elementIds: meshData.elements.slice(elementsPerBody * 2).map(e => e.id),
      visible: true,
      color: '#a0aec0'
    }
  ];
};

// Export the main sample data
export const sampleMeshData = generateSampleBeamMesh();
export const sampleMaterials = generateSampleMaterials(sampleMeshData);
export const sampleBodies = generateSampleBodies(sampleMeshData);

// Re-export bracket project data for easy access
export { bracketSampleProject, type BracketProject } from './bracketProjectData';
import 'leaflet';

declare module 'leaflet' {
  namespace TileLayer {
    interface Options {
      name?: string;
    }
  }

  namespace Polyline {
    interface Options {
      isField?: boolean;
    }
  }

  namespace CircleMarker {
    interface Options {
      isHealthMarker?: boolean;
    }
  }
}

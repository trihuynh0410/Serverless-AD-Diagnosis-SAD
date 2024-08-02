<template>
  <div ref="canvasContainer" style="width: 100%; height: 500px;"></div>
</template>

<script>
import { ref, onMounted, watch, nextTick } from 'vue';

function isWebGL2Supported() {
  try {
    const canvas = document.createElement('canvas');
    return !!(window.WebGL2RenderingContext && canvas.getContext('webgl2'));
  } catch (e) {
    return false;
  }
}

export default {
  props: {
    niftiData: {
      type: Object,
      required: true
    }
  },
  setup(props) {
    const canvasContainer = ref(null);
    let nv;

    onMounted(async () => {
      await nextTick();
      if (!isWebGL2Supported()) {
        console.error('WebGL2 is not supported in this browser');
        return;
      }
      if (canvasContainer.value) {
        const canvas = document.createElement('canvas');
        canvas.width = canvasContainer.value.clientWidth;
        canvas.height = canvasContainer.value.clientHeight;
        canvasContainer.value.appendChild(canvas);

        try {
          const { Niivue } = await import('@niivue/niivue');
          nv = new Niivue({
            logging: true,
            dragAndDropEnabled: false,
            backColor: [0, 0, 0, 1],
          });

          await nv.attachToCanvas(canvas);
          console.log('NiiVue attached to canvas successfully');
          if (props.niftiData) {
            await loadNiftiData(props.niftiData);
          }
        } catch (error) {
          console.error('Error initializing or attaching NiiVue:', error);
        }
      }
    });

    watch(() => props.niftiData, async (newData) => {
      if (newData && nv) {
        await loadNiftiData(newData);
      }
    });

    async function loadNiftiData(data) {
  if (!nv) {
    console.error('NiiVue not initialized');
    return;
  }
  try {
    console.log('Data received in loadNiftiData:', data);
    console.log('Data type:', Object.prototype.toString.call(data));
    
    if (!(data instanceof ArrayBuffer)) {
      console.error('Data is not an ArrayBuffer');
      return;
    }

    console.log('Attempting to load volume from ArrayBuffer...');
    await nv.loadFromArrayBuffer(data, 'volume.nii');
    console.log('Volume loaded successfully');

    console.log('Setting slice type...');
    nv.setSliceType(nv.sliceTypeMultiplanar);
    console.log('Slice type set');

    console.log('Drawing scene...');
    nv.drawScene();
    console.log('Scene drawn');
  } catch (error) {
    console.error('Error in loadNiftiData:', error);
    console.error('Error stack:', error.stack);
  }
}

    return { canvasContainer };
  }
}
</script>
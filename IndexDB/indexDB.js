
// Whenever I want to access or modify a DB I need to start a new transaction
const request = window.indexedDB.open('GalleryCategoryData', 1);

request.onerror = (e) => {
    console.error('Database error:', e.target.errorCode);
}
request.onsuccess = function (e) {
    const db = e.target.result;
    console.log('Database opened successfully: ', db);

    // Start a new transaction to add records
    const transaction = db.transaction(['VideoGameImages'], 'readwrite');
    const objectStore = transaction.objectStore('VideoGameImages');

    // Retrieve all records from the object store
    const getAllRequest = objectStore.getAll();

    getAllRequest.onsuccess = function () {
        if (getAllRequest.result.length == 0) {

        }
        console.log('All records retrieved:', getAllRequest.result.length);
    };

    getAllRequest.onerror = function (event) {
        console.error('Error retrieving all records:', event.target.error);
    };



    // Caching all images to IndexedDB
    // data.VideoGameImages.forEach(img => {
    //     // Cache
    //     const addRequest = objectStore.add({ Image: img });
    //     addRequest.onsuccess = () => console.log('Image record added successfully:', img);
    //     addRequest.onerror = (event) => console.error('Error adding image record:', event.target.error);
    // });

    // transaction.oncomplete = () => console.log('All records added successfully.');
    // transaction.onerror = (event) => console.error('Transaction error:', event.target.error);
};

request.onerror = function (event) {
    console.error('Error opening database:', event.target.error);
};

// Whenever I want to create an object store (Table) do it here
request.onupgradeneeded = (e) => {
    const db = e.target.result;
    // Create an object store (Table) named 'GalleryData' with 'id' as the keyPath
    if (!db.objectStoreNames.contains('VideoGameImages')) {
        // When you create an object store (Table), you only define the keyPath and autoIncrement options.
        const objectStore = db.createObjectStore('VideoGameImages', {
            keyPath: 'id',
            autoIncrement: true
        });

        // So essentially the 'Image' property is the column in the schema that the indexKey (imageIndex) will reference.
        objectStore.createIndex('imageIndex', 'Image', { unique: false });
    }
}